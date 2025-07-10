class FwdContext:
    cum_token_size_list = None

    @staticmethod
    def set_cum_token_size_list(size_list):
        FwdContext.cum_token_size_list = size_list

    @staticmethod
    def get_cum_token_size_list():
        return FwdContext.cum_token_size_list
