package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersNetwork;

public class EmbeddingLayer extends Layer {

	SimpleMatrix embeddings;
	int vocabSize;
	int[] lastInputs;
	boolean masking;
	final SimpleMatrix none;
	
	public EmbeddingLayer(int inputs, int embeddingDepth, int vocabSize, boolean masking) {
		super(inputs, inputs);
		this.vocabSize = vocabSize;
		this.depth = embeddingDepth;
		setGradientSize(inputs, embeddingDepth);
		embeddings = new SimpleMatrix(new float[vocabSize][embeddingDepth]);
		lastInputs = new int[inputs];
		this.masking = masking;
		none = SimpleMatrix.filled(1, embeddingDepth, 0);
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
		none = SimpleMatrix.filled(1, embeddingDepth, 0);
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
			lastInputs[i] = embedding;
			///*
			if(masking)
			{
				if(embedding == -1)
				{
					lastActivation.setRow(i, none);
					masks[i] = true;
					continue;
				}
				masks[i] = false;
			}//*/
			else {
				if(embedding == -1)
				{
					lastActivation.setRow(i, none);
					continue;
				}
			}
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
			if(lastInputs[i] == -1)
				continue;
			embeddings.setRow(lastInputs[i], embeddings.getRow(lastInputs[i]).minus(nextErrorWeighted.getRow(i).scale(model.getLearningRate())));
		}
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
	
	@Override
	public String stringify() {
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

	public static EmbeddingLayer load(String string, LayersNetwork model, int position) {
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
