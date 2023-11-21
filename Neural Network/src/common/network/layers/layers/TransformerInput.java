package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

/**
 * A layer representing the input stack to the Encoder/Decoder of a Transformer. Consists of an {@link InputLayer},
 * an {@link EmbeddingLayer}, and a {@link PositionalEncoding} layer.
 * @author C. Cooper
 */
public class TransformerInput extends Layer {

	final InputLayer input;//The InputLayer
	final EmbeddingLayer embedding;//The EmbeddingLayer
	final PositionalEncoding positionalEncoding;//The PositionalEncoding layer.
	
	/**
	 * Creates a TransformerInput whose input dimension is the given sequenceLength, whose embedding depth
	 * is the given embeddingDepth, and whose encoder's vocab size is the given vocabSize.
	 * @param sequenceLength The dimensions of the input to this layer.
	 * @param embeddingDepth The size of the embedding vectors.
	 * @param vocabSize The number of tokens in the vocab.
	 */
	public TransformerInput(int sequenceLength, int embeddingDepth, int vocabSize)
	{
		super(sequenceLength, sequenceLength);//Super constructor w/ inputs = outputs = sequenceLength
		depth = embeddingDepth;//Sets this layer's depth to the embed depth.
		
		input = new InputLayer(sequenceLength);//Creates an InputLayer with size sequenceLength
		embedding = new EmbeddingLayer(input, embeddingDepth, vocabSize, true);//Creates an embed layer with the given embed depth and vocab size. 
		positionalEncoding = new PositionalEncoding(embedding);//Adds positional encoding to the embed layer.
	}
	
	public TransformerInput(int sequenceLength, int[] embeddingDepths, int totalEmbeddingDepth, int[] vocabSizes)
	{
		super(sequenceLength, sequenceLength);
		depth = totalEmbeddingDepth;
		
		input = new InputLayer(sequenceLength, vocabSizes.length);
		embedding = (embeddingDepths == null && vocabSizes.length == 1) ? new EmbeddingLayer(input, totalEmbeddingDepth, vocabSizes[0], true) : new EmbeddingLayer2D(input, embeddingDepths, totalEmbeddingDepth, vocabSizes, true);
		positionalEncoding = new PositionalEncoding(embedding);
	}
	
	/**
	 * More low-level constructor used in the {@link #load(String, LayersModel, int)} function.
	 * @param inputLayer This layer's internal InputLayer
	 * @param embedding This layer's internal EmbeddingLayer
	 * @param positionalEncoding This layer's internal PositionalEncoding layer.
	 */
	private TransformerInput(InputLayer inputLayer, EmbeddingLayer embedding, PositionalEncoding positionalEncoding)
	{
		super(embedding.inputs, embedding.inputs);
		depth = embedding.depth;
		
		this.input = inputLayer;
		this.embedding = embedding;
		this.positionalEncoding = positionalEncoding;
	}
	
	/**
	 * Takes the activation of this layer. Unlike most layers, the input param is actually used because this layer
	 * has no preceding layer.
	 * @param input The input to this layer.
	 * @param isInference Whether this activation is for inference (no effect).
	 * @return The activation of this layer.
	 */
	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		this.input.activation(input, isInference);//Activates the input layer with the given input
		embedding.activation(null, isInference);//Activates the embedding layer
		lastActivation = positionalEncoding.activation(null, isInference);//Activates the positional encoding layer
		return lastActivation;
	}

	@Override
	public void backprop() {
		//Unlike most layers, this layer doesn't use getGradient() because it just passes gradients sent to it directly to the 
		
		embedding.backprop();//Just backprops the embed layer (the other two don't learn).
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
		return getId() + " " + input.getId() + " " + positionalEncoding.getId() + " " + inputs + " " + (embedding instanceof EmbeddingLayer2D) + "\n" + embedding.stringify();
	}
	
	public static TransformerInput load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int inputId = scanner.nextInt();
		int positionalId = scanner.nextInt();
		int inputs = scanner.nextInt();
		boolean is2D = scanner.nextBoolean();
		StringBuilder builder = new StringBuilder();
		while(scanner.hasNextLine())
			builder.append(scanner.nextLine() + "\n");
		scanner.close();
		InputLayer inputLayer = new InputLayer(inputs);
		inputLayer.setId(inputId);
		inputLayer.setModel(model);
		EmbeddingLayer embedding = is2D ? EmbeddingLayer2D.load(builder.toString(), model) : EmbeddingLayer.load(builder.toString(), model, -1);
		embedding.setModel(model);
		PositionalEncoding encoding = new PositionalEncoding(embedding);
		encoding.setId(positionalId);
		encoding.setModel(model);
		TransformerInput out = new TransformerInput(inputLayer, embedding, encoding);
		out.setId(id);
		return out;
	}
}
