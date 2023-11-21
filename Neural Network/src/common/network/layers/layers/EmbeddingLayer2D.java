package common.network.layers.layers;

import java.util.Arrays;
import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

/**
 * An {@link EmbeddingLayer} that allows 2d inputs. 
 * @author C. Cooper
 */
public class EmbeddingLayer2D extends EmbeddingLayer {

	private final SimpleMatrix[][] embedVectors;
	private final int[] vocabSizes;
	private final int[] embedStartingPositions;
	private final int depthCount;
	private final SimpleMatrix[] nones;
	
	private int lastInputs[][];
	
	/**
	 * @param last
	 * @param embeddingDepth
	 * @param vocabSize
	 * @param masking
	 */
	public EmbeddingLayer2D(Layer last, int[] embeddingDepths, int totalEmbedDepth, int[] vocabSizes, boolean masking) {
		super(last, totalEmbedDepth, 0, masking);
		depthCount = vocabSizes.length;
		
		if(embeddingDepths == null)
		{
			if(totalEmbedDepth % last.depth == 0)
			{
				embeddingDepths = new int[last.depth];
				Arrays.fill(embeddingDepths, totalEmbedDepth / last.depth);
			}else {
				throw new IllegalArgumentException("If you don't provide custom embed depths, the total embed depth must be a multiple of the previous layer's depth.");
			}
		}
		this.vocabSizes = vocabSizes;
		embedStartingPositions = new int[depthCount+1];
		nones = new SimpleMatrix[depthCount];
		embedVectors = new SimpleMatrix[depthCount][];
		lastInputs = new int[inputs][totalEmbedDepth];
		
		int currentStart = 0;
		for(int i = 0; i < depthCount; i++)
		{
			embedStartingPositions[i] = currentStart;
			currentStart += embeddingDepths[i];
			embedVectors[i] = new SimpleMatrix[vocabSizes[i]];
			for(int j = 0; j < vocabSizes[i]; j++)
			{
				embedVectors[i][j] = initMatrix(embeddingDepths[i]);
			}
			nones[i] = new SimpleMatrix(1, embeddingDepths[i]);
		}
		embedStartingPositions[depthCount] = currentStart;
		
		if(currentStart != totalEmbedDepth)
			throw new IllegalArgumentException("The sum of the embedding depths must equal the total embed depth!");
	}
	
	private EmbeddingLayer2D(Layer last, int[] embeddingDepths, int totalEmbedDepth, int[] vocabSizes, boolean masking, SimpleMatrix[][] vectors)
	{
		super(last, totalEmbedDepth, 0, masking);
		
		depthCount = vocabSizes.length;
		
		if(embeddingDepths == null)
		{
			if(totalEmbedDepth % last.depth == 0)
			{
				embeddingDepths = new int[last.depth];
				Arrays.fill(embeddingDepths, totalEmbedDepth / last.depth);
			}else {
				throw new IllegalArgumentException("If you don't provide custom embed depths, the total embed depth must be a multiple of the previous layer's depth.");
			}
		}
		this.vocabSizes = vocabSizes;
		embedStartingPositions = new int[depthCount+1];
		nones = new SimpleMatrix[depthCount];
		embedVectors = vectors;
		lastInputs = new int[inputs][depthCount];
		
		int currentStart = 0;
		for(int i = 0; i < depthCount; i++)
		{
			embedStartingPositions[i] = currentStart;
			currentStart += embeddingDepths[i];
			nones[i] = new SimpleMatrix(1, embeddingDepths[i]);
		}
		embedStartingPositions[depthCount] = currentStart;
		
		if(currentStart != totalEmbedDepth)
			throw new IllegalArgumentException("The sum of the embedding depths must equal the total embed depth!");
	}

	private static SimpleMatrix initMatrix(int vextorSize)
	{
		return SimpleMatrix.random(1, vextorSize).minus(0.5).scale(2);
	}
	
	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
		input = lastLayer.getLastActivation();//Gets the last layer's activation
		//input.print();
		
		inputsLoop: for(int pos = 0; pos < depthCount; pos++)
		{
			depthLoop: for(int i = 0; i < inputs; i++)//For each token in the input
			{
				int embedding = (int)input.get(i, pos);//Casts it to an int to get the token
				lastInputs[i][pos] = embedding;//Adds the token to the last inputs
				
				if(embedding == -1)//If this is a padding token,
				{
					lastActivation.insertIntoThis(i, embedStartingPositions[pos], nones[pos]);
					if(masking)//If we're doing masking,
					{
						masks[i] = true;//Sets the masks at this position to true
					}
					continue depthLoop;
				}
				
				masks[i] = false;//Sets to mask i to false.
				
				if(embedding < 0 || embedding >= vocabSizes[pos])//If the token is out of the vocab,
				{
					throw new IndexOutOfBoundsException("Embedding index is out of range!");
				}
				lastActivation.insertIntoThis(i, embedStartingPositions[pos], embedVectors[pos][embedding]);
			}
		}
		
		return lastActivation;
	}
	
	@Override
	public void backprop() {
		SimpleMatrix nextErrorWeighted = getGradient().scale(model.getLearningRate());//Gets the gradient.
		clearGradients();
		
		for(int i = 0; i < inputs; i++)//For each input,
		{
			for(int j = 0; j < depthCount; j++)
			{
				if(lastInputs[i][j] == -1)//If its token was -1, skips it.
					continue;
				
				/*
				 * Subtracts the error corresponding to this input from the embedding vector of the token for this input.
				 */
				embedVectors[j][lastInputs[i][j]].minus(nextErrorWeighted.extractMatrix(i, i+1, embedStartingPositions[j], embedStartingPositions[j+1]));
			}
		}
	}
	
	@Override
	public String name() {
		return "Embedding2D";
	}
	
	@Override
	public String className() {
		return "Embedding2D";
	}
	
	@Override
	public String stringify() {
		StringBuilder builder = new StringBuilder(getId() + " " + lastLayer.getId() + " " + vocabSizes.length + " " + embedStartingPositions[embedStartingPositions.length-1] + "\n");
		for(int i = 0; i < vocabSizes.length; i++)
		{
			builder.append(vocabSizes[i]);
			builder.append(" ");
			builder.append(embedVectors[i][0].getNumCols());
			builder.append(" ");
			for(int j = 0; j < vocabSizes[i]; j++)
			{
				for(int k = 0; k < embedVectors[i][j].getNumCols(); k++)
				{
					builder.append(embedVectors[i][j].get(k));
					builder.append(" ");
				}
			}
			builder.append("\n");
		}
		return builder.toString();
	}
	
	public static EmbeddingLayer2D load(String layer, LayersModel model)
	{
		Scanner scanner = new Scanner(layer);
		int id = scanner.nextInt();
		int lastId = scanner.nextInt();
		int depthCount = scanner.nextInt();
		int totalEmbedDepth = scanner.nextInt();
		int[] vocabSizes = new int[depthCount];
		int[] embedDepths = new int[depthCount];
		Layer last = model.getLayerByID(lastId);
		SimpleMatrix[][] embedVectors = new SimpleMatrix[depthCount][];
		for(int i = 0; i < depthCount; i++)
		{
			vocabSizes[i] = scanner.nextInt();
			embedDepths[i] = scanner.nextInt();
			embedVectors[i] = new SimpleMatrix[vocabSizes[i]];
			for(int j = 0; j < vocabSizes[i]; j++)
			{
				embedVectors[i][j] = new SimpleMatrix(1, embedDepths[i]);
				for(int k = 0; k < embedDepths[i]; k++)
				{
					embedVectors[i][j].set(k, scanner.nextDouble());
				}
			}
		}
		scanner.close();
		
		EmbeddingLayer2D out = new EmbeddingLayer2D(last, embedDepths, totalEmbedDepth, vocabSizes, true, embedVectors);
		out.setId(id);
		return out;
	}
}
