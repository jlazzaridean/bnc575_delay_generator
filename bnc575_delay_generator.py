import serial
import warnings
import time
import numpy as np
from math import isclose, floor, ceil

class Delay_Generator:
    """
    Device adapter for the BNC 575 delay generator from Berkeley
    Nucleonics Corporation. Functionality exists beyond what is
    implemented here (in particular note that triggering is implemented
    but gating is not.)

    Usage note: calling configure_system before configure_channel will
    result in more robust operation. The configure_channel checks for
    sensibility of certain values given the system settings but the
    configure_system function does not check channel settings.
    """
    def __init__(self,
                 which_port,
                 verbose=True,
                 very_verbose=False,
                 name='BNC575',
                 ch_labels=None):
        self.verbose = verbose
        self.very_verbose = very_verbose
        self.n = name
        if self.verbose: print('%s: Initializing delay generator'%self.n)
        try:
            self.port = serial.Serial(port=which_port, baudrate=38400,
                                      timeout=1)
        except serial.serialutil.SerialException:
            raise IOError('No connection to %s delay generator on port %s'
                          %(self.n, which_port))
        time.sleep(0.1) # if we don't wait a bit, reset will hang on 1st boot
        self.reset()
        # set up a dictionary of dictionaries to store the timing
        # parameters for each channel (system is ch0)
        self.params = {0: {'label': 'system',
                           'mode': self.get_system_mode(),
                           'period_s': self.get_system_clock_period(),
                           'trigger_mode': self.get_trigger_mode(),
                           'trigger_level': self.get_trigger_level(),
                           'trigger_logic': self.get_trigger_logic(),
                           'burst_length': self.get_system_burst_length()}}
        self.set_system_is_active(active=False)
        # dict that relates letter codes to numbers
        self.names_to_channels = {'A': 1, 'B': 2, 'C': 3, 'D': 4,
                                  'E': 5, 'F': 6, 'G': 7, 'H': 8}
        for ch in self.names_to_channels:
            if ch in ch_labels: label = ch_labels[ch]
            elif self.n2c(ch) in ch_labels: label=ch_labels[self.n2c(ch)]
            else: label = ch
            d = {'label': label,
                 'active': self.get_channel_is_active(ch),
                 'mode': self.get_channel_mode(ch),
                 'width_s': self.get_channel_pulse_width(ch),
                 'delay_s': self.get_channel_delay(ch),
                 'polarity': self.get_channel_polarity(ch),
                 'burst_length': self.get_channel_burst_length(ch),
                 'dcyc_on': self.get_channel_duty_cycle_on(ch),
                 'dcyc_off': self.get_channel_duty_cycle_off(ch),
                 'mux_list': self.get_channel_multiplexing(ch)}
            self.params[self.n2c(ch)] = d
        for ch in self.names_to_channels.values():
            self.set_channel_is_active(ch, active=False) # all off
        # some plotting defaults for predicting output
        self.plot_time_px = 5e-9 # resolution on plotting output
        self.plot_n_unique_cycles = 2
        self.predict_output(save_fig=False, show_fig=False)
        if verbose:
            print('%s: Successfully initialized'%self.n)

    def n2c(self, name_or_channel):
        if name_or_channel in self.names_to_channels.values():
            return name_or_channel
        elif name_or_channel in self.names_to_channels:
            return self.names_to_channels[name_or_channel]
        else:
            raise ValueError('Invalid %s output channel: %s'
                             %(self.n, name_or_channel))

    def update_pulse_sequence(self,
                              new_params,  # dict of dicts
                              recalc_output=True): # update output predictions
        t0 = time.perf_counter()
        if 0 in new_params:
            self.configure_system(**new_params[0])
        t1 = time.perf_counter()
        for ch, d in new_params.items():
            if ch != 0: # we already applied the system params
                self.configure_channel(ch, **d)
        if self.verbose: print('%s: Loaded new parameters' % self.n)

    def configure_system(self,
                         mode=None,         # normal/single/burst
                         period_s=None,     # 200e-9 <= (float) <=1e3
                         trig_mode=None,    # disabled/trigger/gate/rearm
                         trig_level_V=None, # 0.2<=(float)<=15, round to 0.01
                         trig_logic=None,   # rising/falling
                         burst_length=None):# burst pulses, 1<=(int)<=1e6
        ''' Update system-level parameters. Note that this function
        does NOT automatically update the predicted output array; you
        must use "load_parameters" or call predict_current_output()
        after calling this fxn'''
        if mode is not None:
            self.set_system_mode(mode)
        if period_s is not None:
            self.set_system_clock_period(period_s)
        if trig_mode is not None:
            self.set_trigger_mode(trig_mode)
        if trig_level_V is not None:
            if self.get_trigger_mode() == 'disabled':
                warnings.warn('Setting trigger level; trigger is not enabled.')
            self.set_trigger_level(round(trig_level_V, 2))
        if trig_logic is not None:
            if self.get_trigger_mode() == 'disabled':
                warnings.warn('Setting trigger edge; trigger is not enabled.')
            self.set_trigger_logic(trig_logic)
        if burst_length is not None:
            if self.get_system_mode() != 'burst':
                warnings.warn(
                    'Setting system burst length, but system '
                    'is not in burst mode')
            self.set_system_burst_length(burst_length)

    def configure_channel(self, channel,    # [A, B, C, D, E, F, G, H] or 0-7
                          active=None,      # bool, True or False
                          mode=None,        # normal/single/burst/duty cycle
                          width_s=None,     # per pulse, 10e-9<=(float)<=1e3
                          delay_s=None,     # -1000<=(float)<=1000
                          polarity=None,    # normal/complement/inverted
                          burst_length=None,# burst pulses, 1<=(int)<=1e6
                          dcyc_on=None,     # pulses to match, 1<=(int)<=1e6
                          dcyc_off=None,    # pulses to skip, 1<=(int)<=1e6
                          mux_list=None):   # subset of [1,2,3,4,5,6,7,8]
        ''' Update channel-level parameters. Note that this function
        does NOT automatically update the predicted output array; you
        must use "load_parameters" or call predict_current_output()
        after calling this fxn'''
        if active is not None:
            self.set_channel_is_active(channel, active)
        if mode is not None:
            self.set_channel_mode(channel, mode)
        if delay_s is not None or width_s is not None:
            if width_s is None:
                ch_width = self.get_channel_pulse_width(channel)
            else:
                ch_width = width_s
            if delay_s is None:
                ch_delay = self.get_channel_delay(channel)
            else:
                ch_delay = delay_s
            if self.get_channel_mode(channel) != 'single' and self.get_system_mode(
                ) != 'single': # we are firing > 1 shot
                # check that we are not running over a system clock period
                sys_period = self.get_system_clock_period()
                assert sys_period > (ch_width + ch_delay + 30e-9)
            # go ahead and update if timing is legal
            if width_s is not None:
                self.set_channel_pulse_width(channel, width_s)
            if delay_s is not None:
                self.set_channel_delay(channel, delay_s)
        if burst_length is not None:
            assert self.get_channel_mode(channel) == 'burst'
            self.set_channel_burst_length(channel, burst_length)
        if dcyc_on is not None:
            assert self.get_channel_mode(channel) == 'duty cycle'
            self.set_channel_duty_cycle_on(channel, dcyc_on)
        if dcyc_off is not None:
            assert self.get_channel_mode(channel) == 'duty cycle'
            self.set_channel_duty_cycle_off(channel, dcyc_off)
        if mux_list is not None:
            self.set_channel_multiplexing(channel, mux_list)

    def start(self): # wrapper for the non-intuitive "set system is active"
        self.set_system_is_active(True)

    def stop(self): # wrapper for the non-intuitive "set system is active"
        self.set_system_is_active(False)     

    def get_system_is_active(self):
        if self.very_verbose:
            print('%s: Getting system is active flag'%self.n)
        is_active = self._send(':PULS0:STATE?')
        if self.very_verbose:
            print('%s: System is active == %s'%(self.n,is_active))
        return {'0': False, '1': True}[is_active]

    def set_system_is_active(self, active):
        assert active in (True, False)
        value_to_send = {False: 0, True: 1}[active]
        if self.verbose and active:
            print('%s: Beginning system active period'%self.n)
        elif self.verbose and not active:
            print('%s: Ending system active period'%self.n)
        cmd = ':PULS0:STATE %d' % value_to_send
        assert self._send(cmd) == 'ok'
        # We intentionally don't re-check the getter here. With either
        # single or burst mode, the device will switch itself back to
        # not active after it finishes. Sometimes, especially with
        # very_verbose, we don't come back and check until it's too
        # late, leading us to fail on the assert statement.
        if self.very_verbose:
            print('%s: --> Done setting system is active flag'%self.n)

    def get_system_mode(self):
        if self.very_verbose: print('%s: Getting system mode'%self.n)
        mode = self._send(':PULS0:MOD?')
        # duty cycle mode is omitted here even though box is capable of it
        parsed_mode = {'NORM': 'normal', 'SING': 'single',
                       'BURS': 'burst'}[mode]
        if self.very_verbose:
            print('%s: System mode == %s'%(self.n,parsed_mode))
        return parsed_mode

    def set_system_mode(self, mode):
        # duty cycle mode is omitted here even though box is capable of it
        legal_modes = {'normal': 'NORM', 'single': 'SING', 'burst': 'BURS'}
        if mode not in legal_modes:
            raise ValueError(
                '%s: Attempt to set invalid system mode %s' %(self.n,mode))
        if self.very_verbose:
            print('%s: Setting system mode to %s' %(self.n,mode))
        cmd = ':PULS0:MOD %s' % legal_modes[mode]          
        assert self._send(cmd) == 'ok'
        time.sleep(0.05)
        assert self.get_system_mode() == mode
        self.params[0]['mode'] = mode
        if self.very_verbose: print('%s: --> Done setting system mode.'%self.n)        

    def get_system_clock_period(self):
        if self.very_verbose: print('%s: Getting system clock period'%self.n)
        period = self._send(':PULSE0:PER?')
        if self.very_verbose:
            print('%s: System clock period == %s s'%(self.n,period))
        return float(period)

    def set_system_clock_period(self, period):
        if period < 200e-9 or period > 1000:
            raise ValueError(
                '%s: Invalid system clock period %e s'%(self.n,period))
        period = self._round_to_5ns(period)
        if self.very_verbose:
            print('%s: Setting system clock period to %e s'%(self.n,period))
        cmd = ':PULS0:PER %0.9f' % period
        assert self._send(cmd) == 'ok'
        assert round(self.get_system_clock_period(), 9) == period
        self.params[0]['period_s'] = period
        if self.very_verbose:
            print('%s: --> Done setting system clock period.'%self.n)

    def get_trigger_mode(self):
        if self.very_verbose: print('%s: Getting trigger mode'%self.n)
        trigger_mode = self._send(':PULS0:TRIG:MOD?')
        if self.very_verbose: print('%s: Trigger mode == %s'%(
            self.n,trigger_mode))
        return {'DIS': 'disabled', 'TRIG': 'trigger'}[trigger_mode]

    def set_trigger_mode(self, trigger_mode):
        legal_modes = {'disabled': 'DIS', 'trigger': 'TRIG'}
        if trigger_mode not in legal_modes:
            raise ValueError('%s: System trigger mode %s not recognized'%(
                self.n,trigger_mode))
        if self.very_verbose:
            print('%s: Setting trigger mode to %s'%(self.n,trigger_mode))
        cmd = ':PULS0:TRIG:MOD %s' % legal_modes[trigger_mode]
        assert self._send(cmd) == 'ok'
        assert self.get_trigger_mode() == trigger_mode
        self.params[0]['trigger_mode'] = trigger_mode
        if self.very_verbose:
            print('%s: --> Done setting trigger mode.'%self.n)

    def get_trigger_level(self):
        if self.very_verbose: print('%s: Getting trigger level'%self.n)
        trigger_level = self._send(':PULS0:TRIG:LEV?')
        if self.very_verbose:
            print('%s: Trigger level==%sV'%(self.n,trigger_level))
        return float(trigger_level)

    def set_trigger_level(self, trigger_level):
        if trigger_level < 0.2 or trigger_level > 15:
            raise ValueError(
                '%s: Invalid trigger level of %0.9f V'%(self.n,trigger_level))
        if self.very_verbose:
            print('%s: Setting trigger level to %f V'%(self.n,trigger_level))
        cmd = ':PULS0:TRIG:LEV %0.1f' % trigger_level
        assert self._send(cmd) == 'ok'
        assert self.get_trigger_level() == trigger_level
        self.params[0]['trigger_level'] = trigger_level
        if self.very_verbose:
            print('%s: --> Done setting trigger level'%self.n)

    def get_trigger_logic(self):
        if self.very_verbose: print('%s: Getting ext. trigger logic'%self.n)
        trigger_logic = self._send(':PULS0:TRIG:EDGE?') # manual is wrong here
        if self.very_verbose:
            print('%s: Trigger logic == %s'%(self.n,trigger_logic))
        return {'RIS': 'rising', 'FALL': 'falling'}[trigger_logic]

    def set_trigger_logic(self, trigger_logic):
        legal_logic = {'rising': 'RIS', 'falling': 'FALL'}
        if trigger_logic not in legal_logic:
            raise ValueError('%s: External trigger logic %s not recognized'%(
                    self.n,trigger_logic))
        if self.very_verbose:
            print('%s: Setting external trigger logic to %s'%(
                self.n,trigger_logic))
        cmd = ':PULS0:TRIG:EDGE %s' % legal_logic[trigger_logic] # manual wrong
        assert self._send(cmd) == 'ok'
        assert self.get_trigger_logic() == trigger_logic
        self.params[0]['trigger_logic'] = trigger_logic
        if self.very_verbose:
            print('%s: --> Done setting ext. trigger logic'%self.n)

    def get_system_burst_length(self):
        cmd = ':PULS0:BCO?'
        if self.very_verbose:
            print('%s: Getting system burst length'%self.n)
        burst_length = self._send(cmd)
        if self.very_verbose:
            print('%s: System burst length == %s'%(self.n,burst_length))
        return int(burst_length)

    def set_system_burst_length(self, burst_length):
        assert 1 <= burst_length <= 1e6
        burst_length = int(burst_length)
        cmd = ':PULS0:BCO %d' % burst_length
        if self.very_verbose:
            print('%s: Setting system burst length to %d'%(self.n,burst_length))
        assert self._send(cmd) == 'ok'
        assert self.get_system_burst_length() == burst_length
        self.params[0]['burst_length'] = burst_length
        if self.very_verbose:
            print('%s: --> Done setting system burst length'%self.n)
    
    def get_channel_is_active(self, name):
        channel = self.n2c(name)
        if self.very_verbose:
            print('%s: Getting ch. %s is active flag'%(self.n,name))
        is_active = self._send(':PULS%d:STAT?' % channel)
        if self.very_verbose:
            print('%s: Ch. %s is active flag == %s'%(self.n,name,is_active))
        return {'0': False, '1': True}[is_active]

    def set_channel_is_active(self, name, active):
        channel = self.n2c(name)
        assert active in (True, False)
        value_to_send = {False: 0, True: 1}[active]
        if self.very_verbose:
            print('%s: Setting ch %s is active flag to %d'%(self.n,name,active))
        cmd = ':PULS%d:STAT %d' % (channel, value_to_send)
        assert self._send(cmd) == 'ok'
        assert self.get_channel_is_active(channel) == active
        self.params[channel]['active'] = active
        if self.very_verbose:
            print('%s: --> Done setting ch. %s is active flag'%(self.n,name))

    def get_channel_delay(self, name): # all values are in seconds
        channel = self.n2c(name)
        if self.very_verbose: print('%s: Getting ch. %s delay'%(self.n,name))
        cmd = ':PULS%d:DEL?' % channel
        delay = self._send(cmd)
        if self.very_verbose:
            print('%s: Ch. %s delay == %s'%(self.n,name,delay))
        return float(delay)

    def set_channel_delay(self, name, delay):
        channel = self.n2c(name)
        if delay < -1000 or delay > 1000:
            raise ValueError(
                '%s: attempt to set invalid delay %d s on channel %d'
                %(self.n,delay,channel))
        delay = self._round_to_5ns(delay)
        if self.very_verbose:
            print('%s: Setting ch. %s delay to %f'%(self.n,name,delay))
        cmd = ':PULS%d:DEL %0.9f' % (channel, delay)
        assert self._send(cmd) == 'ok'
        assert isclose(self.get_channel_delay(channel), delay, rel_tol=1e-11)
        self.params[channel]['delay_s'] = delay
        if self.very_verbose:
            print('%s: --> Set ch. %s delay'%(self.n,name))

    def get_channel_polarity(self, name):
        channel = self.n2c(name)
        if self.very_verbose: print('%s: Getting ch. %s polarity'%(self.n,name))
        cmd = ':PULS%d:POL?' % channel
        polarity = self._send(cmd)
        if self.very_verbose:
            print('%s: Channel %s polarity == %s'%(self.n,name,polarity))
        return {'NORM': 'normal', 'COMP': 'complement',
                'INV': 'inverted'}[polarity]

    def set_channel_polarity(self, name, polarity):
        channel = self.n2c(name)
        legal_polarities = {'normal': 'NORM', 'complement': 'COMP',
                            'inverted': 'INV'}
        if self.very_verbose:
            print('%s: Setting ch. %s polarity to %s'%(self.n,name,polarity))
        cmd = ':PULS%d:POL %s' % (channel, legal_polarities[polarity])
        assert self._send(cmd) == 'ok'
        assert self.get_channel_polarity(channel) == polarity
        self.params[channel]['polarity'] = polarity
        if self.very_verbose:
            print('%s: --> Done setting ch. %s polarity'%(self.n,name))

    def get_channel_pulse_width(self, name): # all values are in seconds
        channel = self.n2c(name)
        if self.very_verbose:
            print('%s: Getting ch. %s pulse width'%(self.n,name))
        cmd = ':PULS%d:WIDT?' % channel
        width = self._send(cmd)
        if self.very_verbose:
            print('%s: Ch. %s pulse width == %s s'%(self.n,name,width))
        return float(width)

    def set_channel_pulse_width(self, name, width):
        channel = self.n2c(name)
        if width < 1e-8 or width > 1000:
            raise ValueError('%s: invalid pulse width %d s on ch %d'
                %(self.n,width,channel))
        width = self._round_to_5ns(width)
        if self.very_verbose:
            print('%s: Setting ch. %s pulse width to %f s'%(self.n,name,width))
        cmd = ':PULS%d:WIDT %0.9f' % (channel, width)
        assert self._send(cmd) == 'ok'
        assert isclose(self.get_channel_pulse_width(channel),
                       width, rel_tol=1e-11)
        self.params[channel]['width_s'] = width
        if self.very_verbose:
            print('%s: --> Done setting ch. %s pulse width'%(self.n,name))

    def get_channel_mode(self, name): 
        channel = self.n2c(name)
        if self.very_verbose: print('%s: Getting ch. %s mode'%(self.n,name))
        cmd = ':PULS%d:CMOD?' % channel            
        mode = self._send(cmd)
        parsed_mode = {'NORM': 'normal', 'SING': 'single',
                       'BURS': 'burst', 'DCYC': 'duty cycle'}[mode]
        if self.very_verbose:
            print('%s: Ch. %s mode == %s'%(self.n,name,parsed_mode))
        return parsed_mode

    def set_channel_mode(self, name, mode):
        channel = self.n2c(name)
        legal_modes = {'normal': 'NORM', 'single': 'SING',
                       'burst': 'BURS', 'duty cycle': 'DCYC'}
        if mode not in legal_modes:
            raise ValueError(
                'Attempt to set invalid mode %s on channel %d'
                % (mode, channel))
        if self.very_verbose:
            print('%s: Setting ch. %s mode to %s'%(self.n,name,mode))
        cmd = ':PULS%d:CMOD %s' % (channel, legal_modes[mode])
        assert self._send(cmd) == 'ok'
        assert self.get_channel_mode(name) == mode
        self.params[channel]['mode'] = mode
        if self.very_verbose:
            print('%s: --> Set ch. %s mode'%(self.n,name))

    def get_channel_burst_length(self, name):
        channel = self.n2c(name)
        cmd = ':PULS%d:BCO?' % channel
        if self.very_verbose:
            print('%s: Getting channel %s burst length'%(self.n,name))
        burst_length = self._send(cmd)
        if self.very_verbose:
            print('%s: == Ch %s burst length %s'%(self.n,name,burst_length))
        return int(burst_length)

    def set_channel_burst_length(self, name, burst_length):
        channel = self.n2c(name)
        assert 1 <= burst_length <= 1e6
        cmd = ':PULS%d:BCO %d' % (channel, burst_length)
        if self.very_verbose:
            print('%s: Setting ch. %s burst length to %d'%(
                self.n,name,burst_length))
        assert self._send(cmd) == 'ok'
        assert self.get_channel_burst_length(name) == burst_length
        self.params[channel]['burst_length'] = burst_length
        if self.very_verbose:
            print('%s: --> Set channel %s burst length'%(self.n,name))

    def get_channel_duty_cycle_on(self, name):
        channel = self.n2c(name)
        cmd = ':PULS%d:PCO?' % channel
        if self.very_verbose:
            print('%s: Getting ch %s # pulses on for duty cycle'%(self.n,name))
        p_on = self._send(cmd)
        if self.very_verbose:
            print('%s: Ch %s pulses on for duty cycle = %s'%(self.n,name,p_on))
        return int(p_on)

    def set_channel_duty_cycle_on(self, name, p_on):
        channel = self.n2c(name)
        assert 1 <= p_on <= 1e6
        cmd = ':PULS%d:PCO %d' % (channel, p_on)
        if self.very_verbose:
            print('%s: Setting ch. %s # pulses on for duty cycle to %d'%(
                self.n,name,p_on))
        assert self._send(cmd) == 'ok'
        assert self.get_channel_duty_cycle_on(name) == p_on
        self.params[channel]['dcyc_on'] = p_on
        if self.very_verbose:
            print('%s: --> Set channel %s # pulses on for duty cycle'%(
                self.n,name))

    def get_channel_duty_cycle_off(self, name):
        channel = self.n2c(name)
        cmd = ':PULS%d:OCO?' % channel
        if self.very_verbose:
            print('%s: Getting ch %s # pulses off for duty cycle'%(self.n,name))
        p_off = self._send(cmd)
        if self.very_verbose:
            print('%s: Ch %s pulses off for duty cycle = %s'%(
                self.n,name,p_off))
        return int(p_off)

    def set_channel_duty_cycle_off(self, name, p_off):
        channel = self.n2c(name)
        assert 1 <= p_off <= 1e6
        cmd = ':PULS%d:OCO %d' % (channel, p_off)
        if self.very_verbose:
            print('%s: Setting ch. %s # pulses off for duty cycle to %d'%(
                self.n,name,p_off))
        assert self._send(cmd) == 'ok'
        assert self.get_channel_duty_cycle_off(name) == p_off
        self.params[channel]['dcyc_off'] = p_off
        if self.very_verbose:
            print('%s: --> Set channel %s # pulses off for duty cycle'%(
                self.n,name))

    def get_channel_multiplexing(self, name):
        channel = self.n2c(name)
        cmd = ':PULS%d:MUX?' % channel
        if self.very_verbose:
            print('%s: Getting ch. %s multiplex list'%(self.n,name))
        mux = self._send(cmd)
        binary = format(int(mux), '08b')
        mux_list = []
        ch_rev = list(self.names_to_channels.values())[::-1]
        for character, ch in zip(binary, ch_rev):
            if character == '1':
                mux_list.append(ch)
        mux_list = sorted(mux_list)
        if self.very_verbose:
            print('%s: == Channel %s multiplex list'%(self.n,name),mux_list)
        return mux_list

    def set_channel_multiplexing(self, name, mux_list):
        channel = self.n2c(name)
        assert set(mux_list).issubset(self.names_to_channels.values())
        if self.very_verbose:
            print('%s: Setting ch %s multiplex list to'%(self.n,name),mux_list)
        mux = 0
        for m in mux_list:
            mux += 2**(m-1)
        cmd = ':PULS%d:MUX %d' % (channel, mux)
        assert self._send(cmd) == 'ok'
        m = self.get_channel_multiplexing(channel)
        assert(sorted(mux_list) == self.get_channel_multiplexing(channel))
        self.params[channel]['mux_list'] = m
        if self.very_verbose:
            print('%s: --> Set ch. %s multiplex list'%(self.n,name))

    def reset(self):
        if self.very_verbose: print('%s: Resetting'%self.n)
        assert self._send('*RST') == 'ok'
        if self.very_verbose: print('%s: --> Successfully reset.'%self.n)

    def force_trigger(self):
        if self.very_verbose:
            print('%s: Sending a software trigger'%self.n)
        assert self._send('*TRG') == 'ok'

    def predict_output(self,
                       param_dict=None, # if None, will do current params
                       show_fig=False, # pop up a blocking figure
                       save_fig=True, # save the output to disk
                       fname='delay_gen_output.png', # output filename
                       s_per_px=None): 
        ''' This function takes a dict of dicts with all of the system
        parameters and generates a voltage array with the time
        resolution defind by self.plot_px_per_period. It also gives the
        user an opportunity to visualize the output, either as a pop-up
        figure or as one saved to disk. The input param_dict must
        contain all of the parameters; if you only have a few new ones,
        you can call _fuse_param_dict() to generate a dict that has a
        few substitutions relative to the existing one.'''
        nuu = self.plot_n_unique_cycles # nickname
        if s_per_px == None: # use default
            s_per_px = self.plot_time_px
        else:
            s_per_px = s_per_px
        import matplotlib.pyplot as plt
        assert nuu >= 1 and int(nuu) == nuu
        if param_dict == None:
            param_dict = self.params
        else:
            if self.verbose:
                print('%s: Predicting output from external param dict'%(self.n)+
                      ' without sanitizing inputs')
        sp = param_dict[0] # get the system dict
        cp = {k: v for k, v in param_dict.items() if k != 0}
        if sp['mode'] in ['normal', 'burst']:
            cyc = 1 # number of system periods to get a unique cycle
            for v in cp.values():
                if v['mode'] == 'duty cycle' and v['active'] == True:
                    d_on = v['dcyc_on']
                    d_off = v['dcyc_off']
                    if cyc < (d_on + d_off): cyc = (d_on + d_off)
                if v['active'] == True:
                    d = v['delay_s']
                    w = v['width_s']
                    assert d+w <= (sp['period_s']+30e-9), 'overrun sys period'
            per = cyc * nuu
            if sp['mode'] == 'burst':
                if sp['burst_length'] < per: per = sp['burst_length']
            px_per_period = int(sp['period_s'] / s_per_px) #period is a multiple
            max_t = sp['period_s'] * per
        elif sp['mode'] == 'single':
            per = 1 # only one system period will execute
            # "system period" doesn't really apply, so we'll take the
            # longest total time on a channel
            max_t = 0
            for v in cp.values():
                if v['active']:
                    t_tot = v['width_s'] + v['delay_s']
                    if t_tot > max_t: max_t = t_tot
            px_per_period = int(max_t / s_per_px) # could truncate weirdly
        # validate we have a sane # of time px and set up the time scale
        if px_per_period > 1e5:
            warnings.warn('%s: Requested %d plot px / period'%(
                self.n,px_per_period))
            s_per_px = s_per_px * px_per_period / 1e5
            px_per_period=int(1e5)
        elif px_per_period == 0:
            raise ValueError('%s: Unable to plot selected s_per_px.'%self.n)
        if self.very_verbose:
            print('%s: Plotting %d px per period'%(self.n, px_per_period))
        t = np.linspace(0, max_t, px_per_period*per)
        a_list = []
        for c, d in cp.items():
            if d['active'] == False:
                a_list.append(np.zeros((px_per_period*per), dtype='bool'))
            else:
                a = np.zeros(px_per_period, dtype='bool')
                delay_px = int(d['delay_s'] / s_per_px)
                width_px = int(d['width_s'] / s_per_px)
                if width_px < 2:
                    warnings.warn('Some features are <2 px for ch %s'%c)
                a[delay_px:(delay_px+width_px)] = 1
                # Channel mode determines how we should tile the array
                if d['mode'] == 'normal':
                    ai = np.tile(a, per)
                elif d['mode'] == 'single':
                    pad = np.zeros(px_per_period*(per-1), dtype='bool')
                    ai = np.concatenate((a, pad))
                elif d['mode'] == 'burst':
                    ch_burst = d['burst_length']
                    if ch_burst >= per:
                        ai = np.tile(a, per)
                    else:
                        aj =  np.tile(a, ch_burst)
                        pad = np.zeros(px_per_period*(per-ch_burst), dtype='bool')
                        ai = np.concatenate((aj, pad))
                elif d['mode'] == 'duty cycle':
                    p_on = d['dcyc_on']
                    p_off = d['dcyc_off']
                    n_full_cycles = int(np.floor(per / (p_on + p_off)))
                    remainder = per % (p_on + p_off)
                    if n_full_cycles >= 1:
                        a_on = np.tile(a, p_on)
                        a_off = np.zeros(px_per_period*p_off, dtype='bool')
                        ac = np.concatenate((a_on, a_off))
                        ad = np.tile(ac, n_full_cycles)
                        if remainder > 0:
                            if remainder <= p_on:
                                ar = np.tile(a, remainder)
                                ai = np.concatenate((ad, ar))
                            else:
                                ar1 = np.tile(a, p_on)
                                ar2 = np.zeros(px_per_period*(remainder-p_on),
                                               dtype='bool')
                                ai = np.concatenate((ad, ar1, ar2))
                        else:
                            ai = ad
                    else:
                        if remainder <= p_on:
                            ai = np.tile(a, remainder)
                        else:
                            ar1 = np.tile(a, p_on)
                            ar2 = np.zeros(px_per_period*(remainder-p_on))
                            ai = np.concatenate((ar1, ar2))                    
                a_list.append(ai)
        v0 = np.array(a_list)
        # To handle multiplexing, we have to now check where the outputs
        # are actually routed
        v = np.zeros((len(cp), px_per_period*per), 'bool')
        for c, d in cp.items():
            if d['active'] == True:
                mux_list = d['mux_list']
                for m in mux_list:
                    v[(c-1), :] += v0[(m-1), :]
        if show_fig or save_fig: # we need to generate a plot
            # convert active ch into rectangles to plot (could plot differently)
            active_ind = np.nonzero(v.sum(axis=1))[0]
            range_list = []
            for i in active_ind:
                currently_on=False
                current_start=-1
                this_list = []
                for j in range(v.shape[1]):
                    if currently_on and v[i,j] == 1:
                        if j == v.shape[1]-1: # ending on, grab this ind
                            end = t[j] - current_start
                            this_list.append((current_start, end))
                            currently_on=False
                        # otherwise, don't do anything
                    elif currently_on and v[i, j] == 0:
                        end = t[j] - current_start
                        this_list.append((current_start, end))
                        currently_on=False
                    elif not currently_on and v[i, j] == 0:
                        continue
                    elif not currently_on and v[i, j] == 1:
                        current_start = t[j]
                        currently_on = True
                range_list.append(this_list)
            h_per_ch = 8; gap=2; width_excess = 30 
            fig, axs = plt.subplots(dpi=300)
            axs.set_aspect(max(t)/h_per_ch/width_excess)
            for i, r in enumerate(range_list):
                axs.broken_barh(r, (i*h_per_ch, h_per_ch-gap),
                                facecolors=plt.cm.tab10.colors[i])
            if sp['mode'] != 'single':
                for i in range(per - 1):
                    axs.axvline(sp['period_s']*(i+1),
                                color='#808080', linestyle='--')
            labels = np.array([cp[k]['label'] for k in cp])
            axs.set_yticks(np.linspace(
                (h_per_ch-gap)/2, h_per_ch*(len(active_ind)-1)+(h_per_ch-gap)/2,
                len(active_ind)), labels=labels[active_ind])
            axs.ticklabel_format(axis='x', scilimits=(0,0))
            axs.set_xlim(min(t), max(t))
            axs.set_ylabel('Channels')
            axs.set_xlabel('Time (s)')
            axs.invert_yaxis()
            if sp['mode'] == 'single':
                axs.set_title('Single shot output')
            elif sp['mode'] == 'normal':
                axs.set_title('%d unique cycle(s) from continuous system play'%
                          self.plot_n_unique_cycles)
            elif sp['mode'] == 'burst' and sp['burst_length'] :
                axs.set_title('%d unique cycle(s) from total burst length %d'%
                              (nuu, sp['burst_length']))
            # Unique cycles are weird in the context of ch burst or single shot
            # Let's at least warn the user about this
            w = [c for c in cp if cp[c]['active']==1 and cp[c]['mode'] in ['burst',
                                                                           'single']]
            if len(w) > 0:
                plt.text(0, -25,
                         'Note: ch(s) in burst or single mode: {}'.format(w))
                plt.text(0, -15, 'Behavior will differ after ch finishes.')
            plt.tight_layout(); plt.grid('on', alpha=0.3)
            if save_fig:
                plt.savefig(fname, bbox_inches='tight')
            if show_fig:
                plt.show()
            plt.close()
        return (t, v) # timescale and voltage array for all ch in param dict

    def _fuse_param_dict(self,
                         new_params, # params to change (dict)
                         old_params=None): # if None, use currently loaded ones
        ''' This function takes new parameters and outputs an array that
        contains current settings where no parameter is specified and a
        new parameter where one was passed. It does NOT update any of
        the settings on the BNC 575; it is intended to be used to
        produce visuals/graphs while selecting settings. Also note: you
        do not need to use this function to generate a complete param
        dict to use the update_pulse_sequence() function, the
        configure_system() function or the configure_channel() function.
        For those, simply pass "None" where no change is desired.'''
        if old_params == None:
            old_params = self.params
        params_to_predict = {}
        for ch in self.params:
            if ch in new_params: # at least some params are changing
                ch_dict = {}
                for k, v in old_params[ch].items(): # extract nested dict
                    if k in new_params[ch] and new_params[ch][k] is not None:
                        ch_dict[k] = new_params[ch][k]
                    else:
                        ch_dict[k] = old_params[ch][k]
                params_to_predict[ch] = ch_dict
            else: # maintain current dict for this ch unchanged
                params_to_predict[ch] = old_params[ch]
        return params_to_predict

    def _round_to_5ns(self, value):
        # rounds floating point values to match precision of 575 (5 ns)
        rounded = int(round(value*1e10))
        if rounded % 50 == 0: # it already ends in 5 or 0
            return rounded * 1e-10
        elif rounded % 100 < 25:
            return floor(rounded/100)*1e-8
        elif rounded % 100 < 75:
            return floor(rounded/100)*1e-8 + 5e-9
        else:
            return ceil(rounded/100)*1e-8

    def _send(self, cmd):
        assert isinstance(cmd, str)
        cmd = bytes(cmd + '\r\n', 'ascii')
        if self.very_verbose:
            print('%s: Bytes written:'%self.n, cmd)
        self.port.write(cmd)
        response = self.port.read_until(expected=b'\r\n')
        if self.very_verbose:
            print('%s: Bytes received:'%self.n, response)
        assert self.port.in_waiting == 0
        return response.decode('ascii').strip('\r\n')

    def close(self):
        self.reset()
        if self.verbose:
            print('%s: Beginning closing protocol...'%self.n)
        self.set_system_is_active(active=False)
        for channel in self.names_to_channels.values():
            self.set_channel_is_active(channel, active=False) # all off
        if self.verbose:
            print('%s: Closing delay generator COM port'%self.n)
        self.port.close()
        if self.verbose:
            print('%s: --> COM port closed.'%self.n)


if __name__ == '__main__':
    delay_gen = Delay_Generator('COM9',
                                verbose=True,
                                very_verbose=False,
                                ch_labels={'A': 'dicam', 'B':'488', 'C':'785',
                                           'D':'830','E':'915', 'F':'940'})
    import matplotlib.pyplot as plt

    # Demonstration 1: a 5 second pulse train
    print('\n\n1. Playing a sequence of pulses for you...')
    sys_params = {'mode': 'normal', 'period_s': 0.1, 'trig_mode': 'disabled'}
    delay_gen.configure_system(**sys_params) # example: unpack the dict
    params = {'polarity': 'normal', 'active': True, 'delay_s': 0.0023,
              'width_s': 0.02} # example of passing a dictionary
    delay_gen.configure_channel('B', **params) # example: unpack the dict
    print('Close plot window to begin pulse train.')
    delay_gen.predict_output(save_fig=True,show_fig=True,fname='dg1.png',
                             s_per_px=1e-6)
    delay_gen.start()
    time.sleep(1)
    delay_gen.stop()

    # Demonstration 2: a short single shot with (mock) external trigger
    print('\n\n2. 10 ns pulse with a software-based mock trigger')
    delay_gen.configure_system(mode='single',
                               trig_mode='trigger',
                               trig_level_V=2.5,
                               trig_logic='rising')
    delay_gen.configure_channel(channel='B',
                                active=True,
                                width_s=10e-9,
                                delay_s=0)
    print('Close plot window to begin pulse train.')
    delay_gen.predict_output(save_fig=True,show_fig=True,fname='dg2.png')
    delay_gen.start()
    delay_gen.force_trigger()
    delay_gen.stop()

    # Demonstration 3: channel burst mode. 2 pulses on B and 1 on C.
    # An example of using the load_pulse_sequence function
    print('\n\n3. Channel single shot and burst mode')
    delay_gen.update_pulse_sequence(
        {0: {'mode': 'normal', 'period_s': 2e-3, 'trig_mode': 'disabled'},
         2: {'mode': 'burst', 'active': True, 'burst_length': 2,
             'width_s': 1e-4, 'delay_s': 0},
         3: {'mode': 'single', 'active': True, 'width_s': 1e-4,
             'delay_s': 1.5e-3}})
    delay_gen.predict_output(save_fig=True, show_fig=True, fname='dg3.png',
                             s_per_px=1e-6)
    delay_gen.start()
    time.sleep(1) # gives the pulse sequence time to run
    delay_gen.stop()
    
    # Demonstration 4: duty cycle mode, with one channel going 1/2 freq
    print('\n\n4. Channel duty cycle with a system burst.')
    new_params = {0: {'mode': 'burst', 'burst_length': 10, 'period_s': 2e-3,
                      'trig_mode': 'disabled'},
                  2: {'mode': 'normal', 'width_s': 1e-4, 'delay_s': 0,
                      'active': 1},
                  3: {'mode': 'duty cycle', 'active': 1, 'width_s': 1e-4,
                      'dcyc_on': 1, 'dcyc_off': 1, 'delay_s': 0}}
    # Example of previewing before sending parameters to the delay generator
    full_set = delay_gen._fuse_param_dict(new_params)
    print('Close plot window to begin pulse train.')
    delay_gen.predict_output(param_dict=full_set, show_fig=1, save_fig=1,
                             fname='dg4.png', s_per_px=1e-6)
    delay_gen.update_pulse_sequence(new_params)
    delay_gen.start()
    time.sleep(0.5) # give pulse sequence time to run
    delay_gen.stop()

    # Demonstration 5: channel multiplexing
    print('\n\n5. Flexibility of channel multiplexing.')
    delay_gen.configure_system(mode='single', trig_mode='trigger',
                               trig_level_V=2.5, trig_logic='rising')
    delay_gen.configure_channel('A', active=True, mode='normal',
                                width_s=1e-4, delay_s=0, mux_list=[1, 2, 3])
    delay_gen.configure_channel('B', active=True, mode='normal',
                                width_s=1e-4, delay_s=1e-3, mux_list=[1, 2])
    delay_gen.configure_channel('C', active=True, mode='normal',
                                width_s=1e-4, delay_s=2e-3)
    print('Close plot window to begin pulse train.')
    delay_gen.predict_output(save_fig=True,show_fig=True,fname='dg5.png',
                             s_per_px=1e-6)
    delay_gen.start()
    delay_gen.force_trigger()
    delay_gen.stop()


    delay_gen.close()
