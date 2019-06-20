# Active Learning

* __data-source__ : a dump of text-files, any kind of read-only database
* __BRAT__ as annotation-frontend
* __postgresql__ as volatile data-management backend
* BRAT writes .ann - files; 
* if special annotation is written by BRAT then the __annotation-collector__ collects it: 
  * push annotation to gitlab repo
  * update postgres??
  
* __predictor__ reads __data-source__, loads a __model__ + writes to volatile-postgres
* __trainer__ reads from annotation-gitrepo and builds a __model__
*