
class CustomSelectionItemBuilder:
    def build(self, market_data, jkp_data):
        return market_data.index.get_level_values("id").unique()
