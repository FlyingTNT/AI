package common.network.layers.layers;

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
		embedding = new EmbeddingLayer(input, embeddingDepth, vocabSize, false);
		positionalEncoding = new PositionalEncoding(embedding);
	}
	
	@Override
	public float[][] activation(float[][] input) {
		this.input.activation(input);
		embedding.activation(null);
		positionalEncoding.activation(input);
		return positionalEncoding.lastActivation;
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
	public void reportGradient(float[][] gradient) {
		embedding.reportGradient(gradient);
	}
	
	@Override
	public float[][] getLastActivation() {
		return positionalEncoding.getLastActivation();
	}
	
	@Override
	public void setModel(LayersNetwork model) {
		super.setModel(model);
		input.setModel(model);
		embedding.setModel(model);
		positionalEncoding.setModel(model);
	}
}
