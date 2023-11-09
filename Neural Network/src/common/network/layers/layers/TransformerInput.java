package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersNetwork;

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
	
	private TransformerInput(EmbeddingLayer embedding)
	{
		super(embedding.inputs, embedding.inputs);
		depth = embedding.depth;
		
		input = new InputLayer(embedding.inputs);
		this.embedding = embedding;
		embedding.lastLayer = input;
		positionalEncoding = new PositionalEncoding(embedding);
	}
	
	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		this.input.activation(input);
		embedding.activation(null);
		positionalEncoding.activation(input);
		return positionalEncoding.getLastActivation();
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
	public void setModel(LayersNetwork model) {
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
		return getId() + " " + inputs + " " + embedding.depth + " " + embedding.vocabSize + "\n" + embedding.stringify();
	}
	
	@Override
	public TransformerInput load(String string, LayersNetwork model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int inputs = scanner.nextInt();
		int depth = scanner.nextInt();
		int vocabSize = scanner.nextInt();
		StringBuilder builder = new StringBuilder();
		while(scanner.hasNextLine())
			builder.append(scanner.nextLine());
		scanner.close();
		EmbeddingLayer embedding = new EmbeddingLayer(0, 0, 0, false).load(builder.toString(), model, -1);
		TransformerInput out = new TransformerInput(embedding);
		out.setId(id);
		return out;
	}
}
