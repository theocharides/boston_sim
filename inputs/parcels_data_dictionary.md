# Parcels Data Dictionary

| Project column | Description |
| --- | --- |
| pid | Unique 10-digit parcel identifier (ward, parcel, and sub-parcel components). |
| condo_id | Condo main parcel ID that groups related condo units. |
| num_bldgs | Number of buildings associated with the parcel record. |
| lu | Land use category/class label. |
| lu_desc | Land use description text. |
| bldg_type | Building type or structural style classification. |
| res_floor | Number of residential floors. |
| condo_res_units | Count of residential units (renamed from assessor RES_UNITS). |
| tt_rooms | Total number of rooms. |
| bed_rms | Number of bedrooms. |
| full_bth | Number of full bathrooms. |
| half_bth | Number of half bathrooms. |
| ktichen | Number/type of kitchens (project field keeps existing typo: ktichen). |
| overall_cond | Overall condition rating. |
| int_condition | Interior condition rating. |
| ext_cond | Exterior condition rating. |
| num_parking | Number of parking spaces. |
| structure_class | Structure class or construction class. |
| yr_remodel | Year of most recent major remodel. |
| yr_built | Year built. |
| land_value | Assessed land value. |
| bldg_value | Assessed building/improvement value. |
| total_value | Total assessed value (land + building + other assessed components). |
| land_sf | Land area in square feet. |
| gross_area | Gross building area. |
| living_area | Living area in square feet. |
| zoning_use | Zoning subdistrict/use designation joined from Boston zoning subdistrict polygons. |
| max_far | Maximum floor area ratio allowed by zoning for the intersecting subdistrict. |
| max_height | Maximum building height allowed by zoning for the intersecting subdistrict. |
| front_setback | Minimum required front setback from the lot line in the intersecting zoning subdistrict. |
| side_setback | Minimum required side setback from the lot line in the intersecting zoning subdistrict. |
| rear_setback | Minimum required rear setback from the lot line in the intersecting zoning subdistrict. |
| max_dua | Maximum dwelling units per area metric allowed by zoning (as provided by source field). |
| max_floors | Maximum number of stories/floors allowed by zoning for the intersecting subdistrict. |