# Parcels Data Dictionary

| Project column | Source column | Description |
| --- | --- | --- |
| pid | PID | Unique 10-digit parcel identifier (ward, parcel, and sub-parcel components). |
| cm_id | CM_ID | Condo main parcel ID that groups related condo units. |
| num_bldgs | NUM_BLDGS | Number of buildings associated with the parcel record. |
| luc | LUC | Land use code (numeric class code). |
| lu | LU | Land use category/class label. |
| bldg_type | BLDG_TYPE | Building type or structural style classification. |
| res_floor | RES_FLOOR | Number of residential floors. |
| cd_floor | CD_FLOOR | Number of condo floors (when applicable). |
| res_units | RES_UNITS | Count of residential units. |
| tt_rooms | TT_RMS | Total number of rooms. |
| bed_rms | BED_RMS | Number of bedrooms. |
| full_bth | FULL_BTH | Number of full bathrooms. |
| half_bth | HLF_BTH | Number of half bathrooms. |
| ktichen | KITCHENS | Number/type of kitchens (project field keeps existing typo: ktichen). |
| overall_cond | OVERALL_COND | Overall condition rating. |
| int_condition | INT_COND | Interior condition rating. |
| ext_cond | EXT_COND | Exterior condition rating. |
| num_parking | NUM_PARKING | Number of parking spaces. |
| structure_class | STRUCTURE_CLASS | Structure class or construction class. |
| yr_remodel | YR_REMODEL | Year of most recent major remodel. |
| yr_built | YR_BUILT | Year built. |
| land_value | LAND_VALUE | Assessed land value. |
| bldg_value | BLDG_VALUE | Assessed building/improvement value. |
| total_value | TOTAL_VALUE | Total assessed value (land + building + other assessed components). |
| land_sf | LAND_SF | Land area in square feet. |
| gross_area | GROSS_AREA | Gross building area. |
| living_area | LIVING_AREA | Living area in square feet. |
| zoning_use | Zoning_Sub | Zoning subdistrict/use designation joined from Boston zoning subdistrict polygons. |
| max_far | Max_FAR | Maximum floor area ratio allowed by zoning for the intersecting subdistrict. |
| max_height | Max_Height | Maximum building height allowed by zoning for the intersecting subdistrict. |
| front_setback | Front_Setb | Minimum required front setback from the lot line in the intersecting zoning subdistrict. |
| side_setback | Side_Setba | Minimum required side setback from the lot line in the intersecting zoning subdistrict. |
| rear_setback | Rear_Setba | Minimum required rear setback from the lot line in the intersecting zoning subdistrict. |
| max_dua | Max_Dwelli | Maximum dwelling units per area metric allowed by zoning (as provided by source field). |
| max_floors | Max_Number | Maximum number of stories/floors allowed by zoning for the intersecting subdistrict. |