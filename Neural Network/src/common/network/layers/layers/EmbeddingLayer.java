package common.network.layers.layers;

import java.util.Arrays;
import org.ejml.simple.SimpleMatrix;

public class EmbeddingLayer extends Layer {

	SimpleMatrix embeddings;
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
		setGradientSize(inputs, embeddingDepth);
		embeddings = new SimpleMatrix(new float[vocabSize][embeddingDepth]);
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
		setGradientSize(inputs, embeddingDepth);
		embeddings = new SimpleMatrix(new float[vocabSize][embeddingDepth]);
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
		lastActivation = new SimpleMatrix(outputs, depth);
		embeddings = SimpleMatrix.random(vocabSize, depth);
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		input = lastLayer.getLastActivation();
		
		//System.out.println("Embedding:");
		//LayersMain.print(embeddings);
		
		for(int i = 0; i < inputs; i++)
		{
			int embedding = (int)input.get(i, 0);
			///*
			if(masking)
			{
				if(embedding == -1)
				{
					lastActivation.setRow(i, embeddings.getRow(0));
					masks[i] = TRUE;
					continue;
				}
				masks[i] = FALSE;
			}//*/
			else {
				if(embedding == -1)
				{
					lastActivation.setRow(i, embeddings.getRow(0));
					continue;
				}
			}
			lastInputs[i] = embedding;
			if(embedding < 0 || embedding >= vocabSize)
			{
				throw new IndexOutOfBoundsException("Embedding index is out of range!");
			}
			lastActivation.setRow(i, embeddings.getRow(embedding));
		}
		return lastActivation;
	}

	@Override//VERIFIED
	public void backprop() {
		SimpleMatrix nextErrorWeighted = getGradient();	
		clearGradients();
		for(int i = 0; i < inputs; i++)
		{
			embeddings.setRow(lastInputs[i], embeddings.getRow(lastInputs[i]).minus(nextErrorWeighted.getRow(i).scale(model.getLearningRate())));
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
