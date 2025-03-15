tf = require("@tensorflow/tfjs-node");
data = require("./data.json");
nlp = require("compromise");

keywords = ["borrower", "mortgator"];
vocab = {
	UNK: 1,
	PAD: 0,
};
index = 2;

function format(str) {
	return str
		.toLowerCase()
		.replace(/[^a-zA-Z0-9\n ]/g, "")
		.split(/[ \n.:,]+/)
		.filter((i) => i);
}

function pad(arr, len) {
	for (i = 0; arr.length < len; i++) {
		arr.push(vocab.PAD);
	}
	return arr;
}

function vocabulize(docs) {
	for (doc of docs) {
		format(doc.text).map((d) => {
			if (!Object.keys(vocab).includes(d)) {
				vocab[d] = index;
				index++;
			}
		});
	}
}

function tokenize(str, padding = 0) {
	return pad(
		format(str).map((v) =>
			Object.keys(vocab).includes(v) ? vocab[v] : vocab.UNK,
		),
		padding,
	);
}

function genMaxLen(docs) {
	l = 0;
	for (d of docs) {
		len = tokenize(d.text).length;
		if (len > l) {
			l = len;
		}
	}
	return l;
}

ML = genMaxLen(data);

function genX(docs, len) {
	return tf.tensor2d(
		docs.map((d) => tokenize(d, len)),
		[docs.length, len],
	);
}

function genY(docs, len) {
	return tf.tensor2d(
		docs.map((d) => {
			x = nlp(format(d.text).join(" ")).json()[0].terms;

			indexes = {};
			x.map((t) => {
				indexes[t.text] = t.index[1];
			});

			tensor = [];

			format(d.entities[0].entity).map((e) => {
				for (i = tensor.length; i < indexes[e]; i++) {
					tensor.push(0);
				}
				tensor.push(1);
			});

			return pad(tensor, len);

			//console.log(x.lookup(format(d.entities[0].entity)))

			//console.log(nlp(format(d.text).join(" ")).match(format(d.entities[0].entity).join(" ")))
		}),
	);
}

function createModel(len) {
	model = tf.sequential();
	model.add(
		tf.layers.embedding({
			inputDim: Object.keys(vocab).length,
			outputDim: 128,
			inputLength: len,
		}),
	);
	model.add(
		tf.layers.lstm({
			units: 64,
			returnSequences: false,
		}),
	);
	model.add(
		tf.layers.dense({
			units: len,
			activation: "sigmoid",
		}),
	);
	model.compile({
		optimizer: "adam",
		loss: "binaryCrossentropy",
		metrics: ["accuracy"],
	});

	return model;
}

function predict(m, doc) {
	fmDoc = format(doc);
	result = m.predict(genX([doc], ML)).arraySync()[0];

	end = [];
	for (const i in result) {
		if (result[i] > 0.15) {
			end.push([result[i], i]);
		}
	}
	end.map((x) => console.log(fmDoc[x[1]] + ": " + x[0]));
}

main = async () => {
	vocabulize(data);
	xTensor = genX(
		data.map((d) => d.text),
		ML,
	);
	yTensor = genY(data, ML);
	model = createModel(ML);

	await model.fit(xTensor, yTensor, {
		epochs: 100,
	});

	await model.save("file://model.json");
	console.log("model saved");
};

if (require.main === module) {
	main();
}

module.exports.predict = predict;
