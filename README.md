# cse151a-trash
## Preprocessing Data

Our main goal is to classify waste items into three general categories: 
* **Landfill**: Items that are non-recyclable or non-compostable and should be disposed of in landfills. 
* **Recyclable**: Items that can be recycled, such as plastics, metals, glass, and paper products.
* **Compost**: Items that can decompose and be used as compost.

As a preprocessing step, we plan on combining the existing 30 waste categories into these broader categories. Additionally, we plan on creating a train test split using ONLY the default image data (comprised of studio images and stock photos of trash) for now. We will also apply min-max normalization to keep our image pixel data on a scale between 0 and 1. After normalizing, we will apply a filter to decrease noise in our images (we haven't yet decided on the best option yet, but it'll likely be either a gaussian or median filter). 
Below is an example of how we may reclassify our data:

| label | category/class |
|:------|:---------------|
| `'aerosol_cans'` | if empty recyclable, otherwise landfill |
| `'aluminum_food_cans'` | recyclable |
| `'aluminum_soda_cans'` | recyclable |
| `'cardboard_boxes'` | recyclable |
| `'cardboard_packaging'` | recyclable |
| `'clothing'` | landfill |
| `'coffee_grounds'` | compost |
| `'disposable_plastic_cutlery'` | landfill |
| `'eggshells'` | compost |
| `'food_waste'` | compost |
| `'glass_beverage_bottles'` | recyclable |
| `'glass_cosmetic_containers'` | recyclable |
| `'glass_food_jars'` | recyclable |
| `'magazines'` | recyclable |
| `'newspaper'` | recyclable |
| `'office_paper'` | recyclable |
| `'paper_cups'` | recyclable as long as not wax coated |
| `'plastic_cup_lids'` | recyclable? |
| `'plastic_detergent_bottles'` | recyclable |
| `'plastic_food_containers'` | recyclable |
| `'plastic_shopping_bags'` | landfill |
| `'plastic_soda_bottles'` | recyclable |
| `'plastic_straws'` | recyclable |
| `'plastic_trash_bags'` | landfill |
| `'plastic_water_bottles'` | recyclable |
| `'shoes'` | landfill |
| `'steel_food_cans'` | recyclable |
| `'styrofoam_cups'` | landfill |
| `'styrofoam_food_containers'` | landfill |
| `'tea_bags'` | compost if plastic free |


