# cse151a-trash
## Preprocessing Data

Our main goal is to classify waste items into three distinct categories: 
* Landfill: Items that are non-recyclable or non-compostable and should be disposed of in landfills. 
* Recyclable: Items that can be recycled, such as plastics, metals, glass, and paper products.
* Compost: Items that can decompose and be used as compost.

As a preprocessing step, we plan on combining the existing 30 waste categories into these broader categories. Additionally, we plan on creating a train test split using ONLY the default image data (comprised of studio images and stock photos of trash) for now. We will also apply min-max normalization to keep our image pixel data on a scale between 0 and 1. After normalizing, we will apply a filter to decrease noise in our images (we haven't yet decided on the best option yet, but it'll likely be either a gaussian or median filter). 



