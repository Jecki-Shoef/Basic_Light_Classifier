# Light Classifier for Agricultural Operations

## The Problem We're Addressing

In modern agriculture, efficient crop management is crucial for maximizing yield and quality. One significant challenge faced by farmers and agricultural researchers is the accurate assessment of plant health and growth stages. This is particularly important in controlled environments like greenhouses or indoor farming setups, where light conditions play a critical role in plant development.

Our Light Classifier addresses a specific aspect of this challenge:

1. **Growth Stage Assessment**: Light conditions often correlate with different growth stages of plants. Accurately classifying these conditions can help in timely interventions and adjustments to growing conditions.

2. **Quality Control**: In large-scale operations, maintaining consistent light conditions across different areas is crucial. This classifier helps in identifying areas that may need adjustment.

## Our Approach to Solving the Problem

We've developed an image classification system that categorizes images based on light conditions. Here's how our solution works:

1. **Image Classification**: We use a machine learning model (Random Forest) trained on a dataset of images labeled as "light" or "dark". This model can classify new images into these categories.

2. **Automated Processing**: The system automatically processes new images placed in a designated input folder.

3. **Result Appending**: After classification, the images are renamed to include their classification and moved to an output folder. This allows for easy sorting and analysis of the results.

4. **User-Friendly Interface**: We've implemented a graphical user interface using Tkinter, making it easy for users to select folders and start the analysis process.


## Benefits of Our Approach

1. **Efficiency**: Automates the process of classifying large numbers of images, saving time and reducing human error.

2. **Consistency**: Provides consistent classification across large datasets, which is crucial for maintaining quality in large-scale operations.

3. **Data Organization**: By automatically sorting and renaming images based on their classification, it simplifies further analysis and record-keeping.

4. **Scalability**: Can be easily scaled to handle larger datasets or more complex classification tasks as needed.

5. **Accessibility**: The graphical interface makes the tool accessible to users without programming experience.

By addressing the challenge of light condition classification, our tool contributes to more efficient and effective agricultural operations, particularly in controlled environment agriculture.
