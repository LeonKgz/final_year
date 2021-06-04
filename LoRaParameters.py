# Time parameters are expressed in ms

class LoRaParameters:
    # Radio wakeup time from SLEEP mode

    RADIO_OSC_STARTUP = 1
    # Radio PLL lock and Mode Ready delay which can vary with the temperature
    RADIO_SLEEP_TO_RX = 2
    # Radio complete Wake-up Time with margin for temperature compensation
    RADIO_WAKEUP_TIME = (RADIO_OSC_STARTUP + RADIO_SLEEP_TO_RX)
    RECEIVE_DELAY1 = 1000
    RECEIVE_DELAY2 = 2000
    JOIN_ACCEPT_DELAY1 = 5000
    JOIN_ACCEPT_DELAY2 = 6000

    RX_WINDOW_1_DELAY = RECEIVE_DELAY1 - RADIO_WAKEUP_TIME
    RX_WINDOW_2_DELAY = RECEIVE_DELAY2 - RADIO_WAKEUP_TIME

    RX_JOIN_WINDOW_1_DELAY = JOIN_ACCEPT_DELAY1 - RADIO_WAKEUP_TIME
    RX_JOIN_WINDOW_2_DELAY = JOIN_ACCEPT_DELAY2 - RADIO_WAKEUP_TIME

    RX_1_NO_ACK_AIR_TIME = [170, 90, 47, 38, 21, 15]  # do not change order (index == DR)
    RX_1_NO_ACK_ENERGY_MJ = [6.4, 3.3, 1.6, 1.3, 0.7, 0.5]  # do not change order (index == DR)

    RX_2_ACK_AIR_TIME = 160
    RX_2_ACK_ENERGY_MJ = 5.6

    RX_2_NO_ACK_AIR_TIME = 40
    RX_2_NO_ACK_ENERGY_MJ = 1.3

    RX_2_DEFAULT_FREQ = 868525000
    RX_2_DEFAULT_SF = 9

    # SPREADING_FACTORS = [12, 11, 10, 9, 8, 7]
    SPREADING_FACTORS = [7, 8, 9, 10, 11, 12]

    SPREADING_FACTORS_COLOURS = {
        7: 'red',
        8: 'blue',
        9: 'green',
        10: 'cyan',
        11: 'magenta',
        12: 'black'
    }

    SPREADING_FACTORS_LABELS = {
        7: 'sf = 7',
        8: 'sf = 8',
        9: 'sf = 9',
        10: 'sf = 10',
        11: 'sf = 11',
        12: 'sf = 12'
    }

    TRANSMISSION_POWERS = [2, 5, 8, 11, 14]

    TRANSMISSION_POWERS_COLOURS = {
        2: 'red',
        5: 'blue',
        8: 'green',
        11: 'cyan',
        14: 'magenta',
    }

    TRANSMISSION_POWERS_LABELS = {
        2: 'tp = 2',
        5:  'tp = 5',
        8:  'tp = 8',
        11: 'tp = 11',
        14: 'tp = 14',
    }

    DEFAULT_CHANNELS = [868100000, 868300000, 868500000]
    CHANNELS = [868100000, 868300000, 868500000, 868525000]
    CHANNEL_DUTY_CYCLE_PROC = {868100000: 1, 868300000: 1, 868500000: 1, 868525000: 10}  # in procent
    CHANNEL_DUTY_CYCLE = {868100000: 1/100, 868300000: 1/100, 868500000: 1/100, 868525000: 10/100}  # not in procent

    JOIN_TX_TIME_MS = 160
    JOIN_TX_ENERGY_MJ = 9

    # Time and Energy consumption of RX1 after join request
    # when a DL message is received
    JOIN_RX_TIME_MS = 120
    JOIN_RX_ENERGY_MJ = 3

    # Time and Energy consumption of RX1 after join request
    # when no DL message is received
    JOIN_RX_1_WINDOW_OPEN_TIME_MS = 26
    JOIN_RX_1_WINDOW_OPEN_ENERGY_MJ = 0.8

    RADIO_TX_PREP_ENERGY_MJ = 0.5  # fixed overhead with each transmission of 500 µJ
    RADIO_TX_PREP_TIME_MS = 40

    # Maximum payload with respect to the datarate index. Cannot operate with repeater.
    MaxPayloadOfDatarate = [51, 51, 51, 115, 242, 242, 242, 242]

    MAX_ACK_RETRIES = 8
    LORAMAC_TX_MIN_DATARATE = 0

    # CR: % 5..8 This is the error correction coding. Higher values mean more overhead.
    # header_implicit_mode -> header is removed
    def __init__(self, freq, sf: int, bw, cr, crc_enabled, de_enabled, header_implicit_mode, tp=14):

        # frequency
        self.freq = freq

        # spreading factor
        self.sf = sf

        # bandwidth
        self.bw = bw

        self.crc = crc_enabled

        # coding rate i.e. => https://en.wikipedia.org/wiki/Code_rate
        self.cr = cr

        # transmission power
        self.tp = tp

        if sf == 7:
            self.dr = 5
        elif sf == 8:
            self.dr = 4
        elif sf == 9:
            self.dr = 3
        elif sf == 10:
            self.dr = 2
        elif sf == 11:
            self.dr = 1
        elif sf == 12:
            self.dr = 0

        if bw == 125 and sf in [11, 12]:
            # low data rate optimization mandated for BW125 with SF11 and SF12
            self.de = 1
        else:
            self.de = de_enabled
        if sf == 6:
            # can only have implicit header with SF6
            self.h = 1
        else:
            self.h = header_implicit_mode

    def change_dr_to(self, dr: int):

        tmp = dr

        if tmp > 5 or tmp < 0:
            raise ValueError('Out of bound DR changing from ' + str(self.dr) + ' to ' + str(tmp))

        self.dr = tmp

        if self.dr == 5:
            self.sf = 7
        elif self.dr == 4:
            self.sf = 8
        elif self.dr == 3:
            self.sf = 9
        elif self.dr == 2:
            self.sf = 10
        elif self.dr == 1:
            self.sf = 11
        elif self.dr == 0:
            self.sf = 12

        if self.bw == 125 and self.sf in [11, 12]:
            # low data rate optimization mandated for BW125 with SF11 and SF12
            self.de = 1
        else:
            self.de = 0
        if self.sf == 6:
            # can only have implicit header with SF6
            self.h = 1
        else:
            self.h = 0

        return self.sf

    def change_tp_to(self, tp: int):
        tmp = tp

        if tmp > 14 or tmp < 2:
            raise ValueError('Out of bound TP changing from ' + str(self.tp) + ' to ' + str(tmp))

        self.tp = tmp
        return self.tp

    def tp_to_index(self, tp: int):
        try:
            return self.TRANSMISSION_POWERS.index(tp)
        except ValueError:
            raise(ValueError(f'The TP entered — {tp}, is not in the list of valid TPs — {self.TRANSMISSION_POWERS}'))

    def change_sf_to(self, sf: int):
        tmp = sf
        if not tmp in self.SPREADING_FACTORS:
            raise ValueError('Out of bound SF changing from ' + str(self.sf) + ' to ' + str(tmp))

        self.sf = tmp

    def sf_to_index(self, sf: int):
        try:
            return self.SPREADING_FACTORS.index(sf)
        except ValueError:
            raise ValueError(f'The SF entered — {sf}, is not in the list of valid SFs — {self.SPREADING_FACTORS}')

    def change_channel_to(self, freq: int):
        if (freq not in self.DEFAULT_CHANNELS):
            raise ValueError('Out of bound channel change from ' + str(self.freq) + ' to ' + str(freq))

        self.freq = freq

    def channel_to_index(self, ch: int):
        try:
            return self.DEFAULT_CHANNELS.index(ch)
        except ValueError:
            raise ValueError(f'The Channel entered — {ch}, is not in the list of default channels — {self.DEFAULT_CHANNELS}')

    def __str__(self):
        return 'SF{}BW{}TP{}'.format(int(self.sf), int(self.bw), int(self.tp))

    @property
    def dr(self):
        # to be sure to return an int
        return int(self.__dr)

    @dr.setter
    def dr(self, dr):
        self.__dr = dr
