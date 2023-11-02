## Dataset Format Description

Taking the mimic dataset as an example, the **directory** description is as follows: 

```
mimic
|   ├── annotation.json
​|   └── images
​			├─ P10
​					├─ p100000XX
​							└─ s5025XXXX
​									├─  7fd9bfbf-619a9611-XXXXXXXX-XXXXXXXX-XXXXXXXX.jpg
​									└─  4427a21f-809986e0-XXXXXXXX-XXXXXXXX-XXXXXXXX.jpg
​					├─  ......
​					└─ p108739XX
​							├─  0.png
​							└─  1.png 
​			├─ p11
​			├─  ......
​			└─  p19
```

Among them, the "annotation.json" file stores a dictionary composed of three parts: "train", "val", and "test". The values are in the form of a list, and each dictionary in the list corresponds to a data sample.

The **data example ** and field description are as follows:

```json
{
	"train": [
		{
            "id": "6e2935c6-0480ce2c-XXXXXXXX-XXXXXXXX-XXXXXXXX", 	#data example ID
            "study_id": 5729XXXX, 
            "subject_id": 1069XXXX, 
            "report": "Frontal and lateral views of the chest.  The heart size and...", 	#medical report
            "image_path": ["p10/p1069XXXX/s5729XXXX/6e2935c6-0480ce2c-XXXXXXXX-XXXXXXXX-XXXXXXXX.jpg"], #image storage path
            "split": "train"
        },
    	...
	],
	"val": [
		{
            "id": "aec4b7d3-55e7e04c-XXXXXXXX-XXXXXXXX-XXXXXXXX", 	#data example ID
            "study_id": 5185XXXX, 
            "subject_id": 1069XXXX, 
            "report": "AP portable view of the chest was obtained.  No focal...", 	#medical report
            "image_path": ["p10/p1069XXXX/s5185XXXX/aec4b7d3-55e7e04c-XXXXXXXX-XXXXXXXX-XXXXXXXX.jpg"], 	#image storage path
            "split": "val"
        },
        ...
    ],
	"test": [
        ...
    ]
}
```
