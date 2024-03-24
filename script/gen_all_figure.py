import facsimlib.plot as facplt

if __name__ == "__main__":

    # General
    facplt.general.lorentz_curve()

    # Rank variation
    facplt.rankvari.rank_variation()
    facplt.rankvari.rank_variation_random_zscore_vs_ratio()

    # Doctorate type
    facplt.doctype.doctorate_group()
    facplt.doctype.lorentz_curve_group()

    facplt.doctype.doctorate_region()
    facplt.doctype.lorentz_curve_region()

    # Hire stats
    facplt.hirestat.hires()

    # Rank moves
    facplt.rankmove.rank_move()
    facplt.rankmove.rank_move_3group_with_inset(net_type='global')
    facplt.rankmove.rank_move_3group_with_inset(net_type='domestic')

    facplt.rankmove.rank_move_region_by_pair(net_type='global')
    facplt.rankmove.rank_move_region_by_pair(net_type='domestic')

