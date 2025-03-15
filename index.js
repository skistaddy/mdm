data = require("./data.json")
tf = require("@tensorflow/tfjs-node")
nlp = require("compromise")

doc = data[0].text


console.log(nlp(doc).out("tags"))
