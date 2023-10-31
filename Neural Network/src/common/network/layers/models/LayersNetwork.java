package common.network.layers.models;

import common.network.layers.Cost;
import common.network.layers.layers.Layer;

public class LayersNetwork {

	protected float learningRate;
	protected Layer[] model;
	protected Cost cost;
	protected int inputs;
	protected int outputs;
	
	public LayersNetwork() {}
	
	public LayersNetwork(float learningRate, Cost cost, Layer... layers) {
		this.learningRate = learningRate;
		this.model = layers;
		this.cost = cost;
		this.inputs = layers[0].inputs;
		this.outputs = layers[layers.length - 1].outputs;
		for(int i = 0; i < layers.length; i++)
		{
			layers[i].setModel(this);
		}
	}

	public void backprop(float[][] input, float[][] targetOutput)
	{
		//FEED FORWARD
		float[][] out = feedForward(input);
		
		//BACKPROP
		float[][] dCostdActivation = cost.derivative(out, targetOutput);

		model[model.length - 1].reportGradient(dCostdActivation);
		for(int i = model.length - 1; i >= 0; i--)
		{
			model[i].backprop();
		}
		return;
	}
	
	public float epoch(float[][][]... trainingSet)
	{
		float avgCost = 0;
		if(trainingSet[0].length != 2)
		{
			throw new IllegalArgumentException("Training set's second dimension must be 2!");
		}
		if(trainingSet[0][0].length != inputs)
		{
			throw new IllegalArgumentException("Training set's inputs must equal the network's input dimension!");
		}
		if(trainingSet[0][1].length != outputs)
		{
			throw new IllegalArgumentException("Training set's outputs must equal the network's output dimension!");
		}
		
		for(int i = 0; i < trainingSet.length; i++)
		{
			model[0].activation(trainingSet[i][0]);
			for(int j = 1; j < model.length; j++)
			{
				model[j].activation(null);
			}
			
			avgCost += cost.cost(model[model.length - 1].getLastActivation(), trainingSet[i][1]);
			float[][] backin = cost.derivative(model[model.length - 1].getLastActivation(), trainingSet[i][1]);
			model[model.length - 1].reportGradient(backin);
			for(int j = model.length - 1; j >= 0; j--)
			{
				model[j].backprop();
			}
		}		
		return avgCost / trainingSet.length;
	}
	
	public float[][] feedForward(float[][] input)
	{
		float[][] out = input;
		for(int i = 0; i < model.length; i++)
		{
			out = model[i].activation(out);
		}
		return out;
	}
	
	public float getLearningRate() {
		return learningRate;
	}
	
	@Override
	public String toString() {
		String out = "Model: " + inputs + " -> " + outputs + "\n";
		for(int i = 0; i < model.length; i++)
		{
			out += model[i] + "\n";
		}
		return out;
	}
}
