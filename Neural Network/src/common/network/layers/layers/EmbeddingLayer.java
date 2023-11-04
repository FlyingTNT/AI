package common.network.layers.layers;

import java.util.Arrays;
import java.util.Random;

import common.network.layers.LayersMain;

public class EmbeddingLayer extends Layer {

	float[][] embeddings;
	int vocabSize;
	int[] lastInputs;
	boolean masking;
	final boolean[] TRUE;
	final boolean[] FALSE;
	final float[] MASK;
	
	public EmbeddingLayer(int inputs, int embeddingDepth, int vocabSize, boolean masking) {
		super(inputs, inputs);
		this.vocabSize = vocabSize;
		this.depth = embeddingDepth;
		embeddings = new float[vocabSize][embeddingDepth];
		lastInputs = new int[inputs];
		this.masking = masking;
		
		FALSE = new boolean[depth];
		boolean[] tr = new boolean[depth];
		Arrays.fill(tr, true);
		TRUE = tr;
		float[] mk = new float[depth];
		Arrays.fill(mk, Float.NEGATIVE_INFINITY);
		MASK = mk;
		init();
	}

	public EmbeddingLayer(Layer last, int embeddingDepth, int vocabSize, boolean masking) {
		super(last, last.outputs);
		this.vocabSize = vocabSize;
		this.depth = embeddingDepth;
		embeddings = new float[vocabSize][embeddingDepth];
		lastInputs = new int[inputs];
		this.masking = masking;
		
		FALSE = new boolean[depth];
		boolean[] tr = new boolean[depth];
		Arrays.fill(tr, true);
		TRUE = tr;
		float[] mk = new float[depth];
		Arrays.fill(mk, Float.NEGATIVE_INFINITY);
		MASK = mk;
		init();
	}
	
	public void init()
	{
		Random random = new Random();
		for(int i = 0; i < vocabSize; i++)
		{
			for(int j = 0; j <  depth; j++)
			{
				embeddings[i][j] = random.nextFloat();
				if(embeddings[i][j] == 0)embeddings[i][j] = 1;
			}
		}
	}

	@Override
	public float[][] activation(float[][] input) {
		input = lastLayer.getLastActivation();
		float[][] out = new float[inputs][depth];
		
		//System.out.println("Embedding:");
		//LayersMain.print(embeddings);
		
		for(int i = 0; i < inputs; i++)
		{
			int embedding = (int)input[i][0];
			lastInputs[i] = embedding;
			///*
			if(masking)
			{
				if(embedding == -1)
				{
					out[i] = embeddings[0];
					masks[i] = TRUE;
					continue;
				}
				masks[i] = FALSE;
			}//*/
			else {
				if(embedding == -1)
				{
					out[i] = embeddings[0];
					continue;
				}
			}
			if(embedding < 0 || embedding >= vocabSize)
			{
				throw new IndexOutOfBoundsException("Embedding index is out of range!");
			}
			out[i] = embeddings[embedding];
		}
		lastActivation = out;
		return out;
	}

	@Override//VERIFIED
	public void backprop() {
		float[][] nextErrorWeighted = getGradient();	
		clearGradients();
		for(int i = 0; i < inputs; i++)
		{
			for(int j = 0; j < depth; j++)
			{
				if(nextErrorWeighted[i][j] > 20 || nextErrorWeighted[i][j] < -20)
				{
					i=i;
				}
				embeddings[lastInputs[i] ][j] -= nextErrorWeighted[i][j] * model.getLearningRate();
			}
		}
		//System.out.println(LayersMain.floatMatrixToString(embeddings, 2));
	}

	@Override
	public String name() {
		return "Embedding";
	}
	
	@Override
	public String toString() {
		return name() + " (" + inputs + ", "+ depth + ")";
	}
	
	public void setMasking(boolean masking) {
		this.masking = masking;
	}
}
