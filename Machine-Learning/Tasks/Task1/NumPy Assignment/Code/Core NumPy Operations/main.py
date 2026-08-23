import numpy as np

grades = [[85, 78, 92, 88],
          [70, 76, 80, 65],
          [90, 88, 94, 91],
          [60, 65, 58, 62],
          [100, 95, 98, 97]]
# Convert the data into a NumPy array
grades_np = np.array(grades)

# Print the shape of the array
print("="*50)
print("The shape of the array:", grades_np.shape)
print("="*50)

#Compute:
# The mean grade of each student
print("The mean grade of each student:")
print(grades_np.mean(axis=1))
print("="*50)

# The mean grade of each subject
print("The mean grade of each subject:")
print(grades_np.mean(axis=0))
print("="*50)

# Extract the students whose average grade is greater than 85
print("Students whose average grade is greater than 85:\n", grades_np[grades_np.mean(axis=1) > 85])
print("="*50)

#Add a bonus of 5 marks to all grades using broadcasting
newGrades = grades_np + 5
print("The new Grades after adding bonus:\n",newGrades)
print("="*50)

#Normalize the grades using Min-Max normalization
normGrades = (newGrades - newGrades.min(axis=0)) / (newGrades.max(axis=0)-newGrades.min(axis=0))
print("The normalized Grades:\n",normGrades)
print("="*50)

#Flatten the array into a single vector
flattenGrades = normGrades.flatten()
print("The flattened Grades:\n",flattenGrades)