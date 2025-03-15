data = require("./model/data.json")
tf = require("@tensorflow/tfjs-node")
nlp = require("compromise")

predict = require("./model.js").predict

main = async () => {
	model = await tf.loadLayersModel("file://model/model.json")
	predict(model, data[0].text)
}

main()




