[SharedTask 2018 Task 3 data](https://healthlanguageprocessing.files.wordpress.com/2018/04/task3_trainingset3_download_form.zip)  
* SMM4H == Social Media Mining 4 Health  
* get the twitter-id-files from [here](https://healthlanguageprocessing.files.wordpress.com/2018/04/task3_trainingset3_download_form.zip)  
* the download-script should produce a jsonl: `wc -l tweets_SMM4H_2018_task3.jsonl` 25605  
`cat tweets_SMM4H_2018_task3.jsonl | jq .text | rg -v 'null' | wc -l`  16145 are non-null  
* can one also srape twitter without having a specific list of ids? 