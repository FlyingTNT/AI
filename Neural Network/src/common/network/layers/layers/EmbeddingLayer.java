package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

/**
 * A basic token embedding layer. Takes the input, casts it to ints, and outputs the embedding vector for each int.
 * @author C. Cooper
 */
public class EmbeddingLayer extends Layer {

	SimpleMatrix embeddings;//The matrix of embeddings (each row is the embedding vector of the token that == its index)
	int vocabSize;//The number of tokens in the vocab (the number of embedding vectors this layer needs)
	int[] lastInputs;//The tokens of the last activation
	boolean masking;//Whether to do masking
	final SimpleMatrix none;//The embed vector for -1 (pad token) (just all zeros)

	/**
	 * Constructor for EmbeddingLayer.
	 * @param last The layer that precedes this layer.
	 * @param embeddingDepth The size of the embedding vectors.
	 * @param vocabSize The number of tokens in the vocab.
	 * @param masking Whether to do masking.
	 */
	public EmbeddingLayer(Layer last, int embeddingDepth, int vocabSize, boolean masking) {
		super(last, last.outputs);
		this.vocabSize = vocabSize;
		this.depth = embeddingDepth;
		setGradientSize(inputs, embeddingDepth);
		embeddings = new SimpleMatrix(vocabSize, embeddingDepth);
		lastInputs = new int[inputs];
		this.masking = masking;
		none = new SimpleMatrix(1, embeddingDepth);
		init();
	}
	
	/**
	 * Initializes the lastActivation and sets the embedding vectors to random numbers -1-1
	 */
	private void init()
	{
		lastActivation = new SimpleMatrix(outputs, depth);
		embeddings = SimpleMatrix.random(vocabSize, depth).minus(0.5).scale(2);//Random is 0-1. -0.5 -> -0.5-0.5, *2 -> -1-1 
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		input = lastLayer.getLastActivation();//Gets the last layer's activation
		
		for(int i = 0; i < inputs; i++)//For each token in the input
		{
			int embedding = (int)input.get(i, 0);//Casts it to an int to get the token
			lastInputs[i] = embedding;//Adds the token to the last inputs
			
			if(embedding == -1)//If this is a padding token,
			{
				lastActivation.setRow(i, none);//Sets the last activation at this position to all zeroes
				if(masking)//If we're doing masking,
					masks[i] = true;//Sets the masks at this position to true
				continue;//Moves on to the next token
			}
			
			masks[i] = false;//Sets to mask i to false.
			
			if(embedding < 0 || embedding >= vocabSize)//If the token is out of the vocab,
			{
				throw new IndexOutOfBoundsException("Embedding index is out of range: " + embedding + "!");
			}
			lastActivation.setRow(i, embeddings.getRow(embedding));//Inserts the embedding vector of the token into the activation matrix.
		}
		return lastActivation;
	}

	@Override
	public void backprop() {
		SimpleMatrix nextErrorWeighted = getGradient();//Gets the gradient.
		clearGradients();
		for(int i = 0; i < inputs; i++)//For each input,
		{
			if(lastInputs[i] == -1)//If its token was -1, skips it.
				continue;
			/*
			 * Subtracts the error corresponding to this input from the embedding vector of the token for this input.
			 */
			embeddings.setRow(lastInputs[i], embeddings.getRow(lastInputs[i]).minus(nextErrorWeighted.getRow(i).scale(model.getLearningRate())));
		}
	}

	@Override
	public String name() {
		return "Embedding";
	}
	
	public void setMasking(boolean masking) {
		this.masking = masking;
	}
	
	@Override
	public String stringify() {
		/*
		 * Returns a string in the form:
		 * thisId LastId inputs embedDepth vocabSize doMasking
		 * token 0 embed vector
		 * token 1 embed vector
		 * ...
		 */
		StringBuilder out = new StringBuilder();
		out.append(getId() + " " + lastLayer.getId() + " " + inputs + " " + depth + " " + vocabSize + " " + masking + "\n");
		for(int i = 0; i < vocabSize; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				out.append(embeddings.get(i, j) + " ");
			}
			out.append("\n");
		}
		return out.toString();
	}

	/**
	 * Loads an EmmbeddingLayer based on a string produced by {@link #stringify()}.
	 * @param string A string produced by {@link #stringify()}.
	 * @param model The model this layer belongs to.
	 * @param position The position of this layer in the model (not used).
	 * @return An AttentionLayer based on the given String.
	 */
	public static EmbeddingLayer load(String string, LayersModel model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int lastId = scanner.nextInt();
		int inputs = scanner.nextInt();
		int depth = scanner.nextInt();
		int vocabSize = scanner.nextInt();
		boolean masking = scanner.nextBoolean();
		EmbeddingLayer out = new EmbeddingLayer(model.getLayerByID(lastId), depth, vocabSize, masking);
		for(int i = 0; i < vocabSize; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				out.embeddings.set(i, j, scanner.nextDouble());
			}
		}
		
		scanner.close();
		
		out.setId(id);
		
		return out;
	}
}
