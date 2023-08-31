This folder contains the data splits from v2 of Social Bias Frames / Social Bias Inference Corpus.
For ease of use, I've included both files with annotations and annotator information (e.g., `"SBIC.v2.trn.csv"`), as well as aggregated per post (e.g., "SBIC.v2.agg.trn.csv").

Simply read in the file using pandas:
```python
df = pd.read_csv("SBIC.v2.trn.csv")
```

Each line in the file contains the following fields (in order):
- _whoTarget_: group vs. individual target
- _intentYN_: was the intent behind the statement to offend
- _sexYN_: is the post a sexual or lewd reference
- _sexReason_: free text explanations of what is sexual
- _offensiveYN_: could the post be offensive to anyone
- _annotatorGender_: gender of the MTurk worker 
- _annotatorMinority_: whether the MTurk worker identifies as a minority
- _sexPhrase_: part of the post that references something sexual
- _speakerMinorityYN_: whether the speaker was part of the same minority group that's being targeted
- _WorkerId_: hashed version of the MTurk workerId
- _HITId_: id that uniquely identifies each post
- _annotatorPolitics_: political leaning of the MTurk worker
- _annotatorRace_: race of the MTurk worker
- _annotatorAge_: age of the MTurk worker
- _post_: post that was annotated
- _targetMinority_: demographic group targeted
- _targetCategory_: high-level category of the demographic group(s) targeted
- _targetStereotype_: implied statement
- _dataSource_: source of the post (`t/...`: means Twitter, `r/...`: means a subreddit)

## Aggregation
the aggregated files were compiled using the following code
```
textFields = ['targetMinority','targetCategory', 'targetStereotype']
classFields = ['whoTarget', 'intentYN', 'sexYN','offensiveYN']

aggDict = {c: lambda x: sorted(filter(lambda x: x, set(x)))
           for c in textFields}

aggDict.update({c: lambda x: np.mean(x) for c in classFields})
df[textFields] = df[textFields].fillna("")

gDf = df.groupby("post",as_index=False).agg(aggDict)
gDf["hasBiasedImplication"] = (gDf["targetStereotype"].apply(len) == 0).astype(int)
gDf[textFields] = gDf[textFields].apply(lambda c: c.apply(json.dumps))
```

To load the list of implications, use `json.loads`

For more information, please see:
Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A Smith, Yejin Choi (2019)
_Social Bias Frames: Reasoning about Social and Power Implications of Language_