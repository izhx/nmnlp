# nmnlp
自用NLP脚手架。
## 一些约定
- view、reshape等操作时，涉及到维度的表示、注释等，必须高维在前。如：`a(n,s,h).view(-1, h) # (n,s,h) -> (n*s,h)`或者`a.view(n*s,h)`。