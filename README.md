# movie_rating_and_award_predition


大学の修了課題として、作ったアカデミー賞を予測するプログラムです。
先生の指示でWeb CrawlingでIMDBからデータを集め、10年分のデータを使って、機械学習を行いました。
Classifier：　Decision Tree（決定木）

集めたデータの中で、ユーザーの評価の内容もありました。
評価内容をNatural Language Natural Language Toolkit (NLKT)で、
評価内容はポジティブかネガティブを分析しました。
分析した結果を点数に変換して、
IMDBの点数とユーザーが評価点数に加えて、予測しました。

アカデミー賞の審査標準がわからないので、3つの点数の比率を変更したり、Test Sizeを変更したり、プログラムの検証をしました。
結果的に、精度は７０％ありました。

成果報告書：https://drive.google.com/file/d/11Fm0Nk5X4FXhK4THUefythJt_rLDzFMf/view?usp=drive_link
＊注：現在は、IMDBのサイトは更新したらしくて、書いたプログラムはもう使えません。
