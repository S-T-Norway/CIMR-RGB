from ap_processing import AntennaPattern


class BGInterp:
    def __init__(self, config):
        self.config = config

    def BG(self):
        pass


    def interp_variable_dict(self, samples_dict, variable_dict, scan_direction=None, band=None):

        ap = AntennaPattern(
            config=self.config,
            band=band
        )

        test = 0














