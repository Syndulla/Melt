# Melt: a Lightweight Faceswap GUI

Melt is a powerful faceswapping tool that allows users to easily swap the faces of multiple videos. Its graphical user interface makes it easy for anyone to use, regardless of their technical expertise. The tool also includes a queue system, which allows users to add multiple videos to be processed in a row. This can save time and effort when working with a large number of videos. In addition to its user-friendly interface and queue system, Melt also utilizes multiprocessing to swap faces in multiple videos simultaneously. This helps to speed up the face swapping process, making it possible to process a large number of videos in a short amount of time. Whether you are a professional video editor or just looking to have some fun with face swapping, Melt is a tool that is worth trying out.

# Please Note

Please be advised that by using this tool, you are solely responsible for your actions and any consequences that may result from your usage. The developers of this tool accept no responsibility for any consequences that may arise from the use of this tool. Use at your own risk and comply with all local regulations.

# Installing Requirements
```
git clone https://github.com/Syndulla/Melt.git
cd  Melt
pip install -r requirements.txt
```
Follow the Insightface [instructions here](https://github.com/deepinsight/insightface/tree/master/examples/in_swapper) to download their pretrained weights.

# Usage
Just navigate to the root directory and run:
```
python melt_gui.py
```

# Examples
<div>
<img width=21% src="./source_images/halle.jpg"/>
<img width=35% src="./examples/halle_source.webp"/>
<img width=35% src="./examples/halle_swap.webp"/>
</div>
<div>
<img width=16% src="./source_images/robert_downey_jr.jpg"/>
<img width=35% src="./examples/robert_source.webp"/>
<img width=35% src="./examples/robert_swap.webp"/>
</div>

# Credits
<!--ts-->
Massive shoutout and credits for the model itself goes to the team at Insightface. They continue to push boundries and deserve to be praised.
* [Insightface Github](https://github.com/deepinsight/insightface)
* [Insightface Home](https://github.com/deepinsight/insightface](https://insightface.ai/)
<!--te-->
