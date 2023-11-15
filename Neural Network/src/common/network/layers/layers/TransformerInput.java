package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

public class TransformerInput extends Layer {

	InputLayer input;
	EmbeddingLayer embedding;
	PositionalEncoding positionalEncoding;
	
	public TransformerInput(int sequenceLength, int embeddingDepth, int vocabSize)
	{
		super(sequenceLength, sequenceLength);
		depth = embeddingDepth;
		
		input = new InputLayer(sequenceLength);
		embedding = new EmbeddingLayer(input, embeddingDepth, vocabSize, true);
		positionalEncoding = new PositionalEncoding(embedding);
	}
	
	private TransformerInput(InputLayer inputLayer, EmbeddingLayer embedding, PositionalEncoding positionalEncoding)
	{
		super(embedding.inputs, embedding.inputs);
		depth = embedding.depth;
		
		this.input = inputLayer;
		this.embedding = embedding;
		this.positionalEncoding = positionalEncoding;
	}
	
	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		this.input.activation(input);
		embedding.activation(null);
		positionalEncoding.activation(null);
		lastActivation = positionalEncoding.getLastActivation();
		return lastActivation;
	}

	@Override
	public void backprop() {
		embedding.backprop();
		input.clearGradients();
	}

	@Override
	public String name() {
		return "Transformer Input";
	}
	
	@Override
	public String className() {
		return "TransformerInput";
	}

	public void setMasking(boolean masking)
	{
		embedding.setMasking(masking);
	}
	
	@Override
	public void reportGradient(SimpleMatrix gradient) {
		embedding.reportGradient(gradient);
	}
	
	@Override
	public SimpleMatrix getLastActivation() {
		return positionalEncoding.getLastActivation();
	}
	
	@Override
	public void setModel(LayersModel model) {
		super.setModel(model);
		input.setModel(model);
		embedding.setModel(model);
		positionalEncoding.setModel(model);
	}
	
	@Override
	public boolean[] getMasks() {
		return positionalEncoding.getMasks();
	}
	
	@Override
	public String stringify() {
		return getId() + " " + input.getId() + " " + positionalEncoding.getId() + " " + inputs + " " + embedding.depth + " " + embedding.vocabSize + "\n" + embedding.stringify();
	}
	
	public static TransformerInput load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int inputId = scanner.nextInt();
		int positionalId = scanner.nextInt();
		int inputs = scanner.nextInt();
		int depth = scanner.nextInt();
		int vocabSize = scanner.nextInt();
		StringBuilder builder = new StringBuilder();
		while(scanner.hasNextLine())
			builder.append(scanner.nextLine() + "\n");
		scanner.close();
		InputLayer inputLayer = new InputLayer(inputs);
		inputLayer.setId(inputId);
		inputLayer.setModel(model);
		EmbeddingLayer embedding = EmbeddingLayer.load(builder.toString(), model, -1);
		embedding.setModel(model);
		PositionalEncoding encoding = new PositionalEncoding(embedding);
		encoding.setId(positionalId);
		encoding.setModel(model);
		TransformerInput out = new TransformerInput(inputLayer, embedding, encoding);
		out.setId(id);
		return out;
	}
}
