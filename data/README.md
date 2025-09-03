## Dataset Format Description

Taking the mimic dataset as an example, the **directory** description is as follows: 

`iu_xray`

​	`├─ annotation.json`

​	`└─ images`

​			`├─ CXR1000_IM-xxxx`

​					`├─  0.png`

​					`└─  1.png` 

​			`├─  ......`

​			`└─ CXR9_IM-xxxx`

​					`├─  0.png`

​					`└─  1.png` 

Among them, the "annotation.json" file stores a dictionary composed of three parts: "train", "val", and "test". The values are in the form of a list, and each dictionary in the list corresponds to a data sample.

The **data example ** and field description are as follows:

```json
{
	"train": [
    	{
        	"id": "CXR2384_IM-0942",	#data example ID
        	"report": "The heart size and pulmonary vascularity ...",	#medical report
        	"image_path": ["CXR2384_IM-0942/0.png", "CXR2384_IM-0942/1.png"],	#image storage path
        	"split": "train"
    	},
    	...
	],
	"val": [
        {
            "id": "CXR2056_IM-0694-1001", 	#data example ID
            "report": "The heart size is upper limits of ...",	#medical report
            "image_path": ["CXR2056_IM-0694-1001/0.png", "CXR2056_IM-0694-1001/1.png"],	#image storage path
            "split": "val"
		},
        ...
    ],
	"test": [
        ...
    ]
}
```

