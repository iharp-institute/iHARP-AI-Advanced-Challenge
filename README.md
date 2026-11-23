# iHARP-AI-Advanced-Challenge
This is a repository to kick-start the iHARP AI Advanced Course Challenge on Englacial Layer Annotation.

Background and Motivation: 

The Greenland and Antarctic ice sheets hold enough ice to raise global sea level by about 7.42 m and 57.9 m respectively. Knowing how fast that ice will reach the ocean depends on knowing how these ice sheets gained snow, flowed and deformed in the past. A key record of that history is stored inside the ice itself.

Airborne ice-penetrating radar, operated by the Center for Remote Sensing and Integrated Systems (CReSIS), produces images called radargrams that show the ice sheet in cross-section. Inside these images are englacial layers: thin, bright lines that trace old snow surfaces buried over thousands of years. Each layer marks ice of roughly the same age. How deep a layer sits tells us how much snow fell, and how it bends and folds tells us how the ice has moved and deformed. Glaciologists use traced layers to date the ice, rebuild past snowfall and test the ice-flow models behind sea-level rise projections.

Turning radargrams into usable layer data is hard:

Manual tracing does not scale. Experts trace layers by hand, one image at a time, while every survey season adds far more images than people can trace.
Automated detectors return broken fragments, not continuous layers. Layers are only a few pixels thick, closely spaced, faint at depth and buried in radar noise, so models tend to output short, disconnected pieces.


The Task

Participants will build/train/finetune AI/ML supervised models that detect englacial layers in Greenland radargrams. You can use (Transfer learning/Few-shot learning/Meta learning, etc). This is a fine-grained curvilinear detection problem: the targets are long, thin, curved lines, background pixels far outnumber layer pixels, and an error of one or two pixels is enough to break a layer.

Your model needs to do two things well:

Detect: for each radargram, output a binary mask that marks every layer pixel.

Stay consistent: where two consecutive radargrams meet, the same layer should continue across the boundary without jumping, splitting or vanishing.

**Note: **Strong solutions will help turn years of radar surveys into continuous, trustworthy layer maps of the Greenland Ice Sheet.
