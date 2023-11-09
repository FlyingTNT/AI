package common.network.layers.models;

import java.util.ArrayList;
import java.util.Scanner;
import org.ejml.simple.SimpleMatrix;

import common.network.layers.Cost;
import common.network.layers.layers.Layer;
import common.network.layers.layers.RotationLayer;
import common.network.layers.layers.StandardLayer;

public class LayersNetwork {

	protected float learningRate;
	protected Layer[] model;
	protected Cost cost;
	protected int inputs;
	protected int outputs;
	private boolean[] masks;
	private ArrayList<Layer> layersList;
	
	public LayersNetwork() {layersList = new ArrayList<>();}
	
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
		layersList = new ArrayList<>();
		
		masks = new boolean[layers[0].inputs];
	}
	
	public float epoch(SimpleMatrix[]... trainingSet)
	{
		float avgCost = 0;
		if(trainingSet[0].length != 2)
		{
			throw new IllegalArgumentException("Training set's second dimension must be 2!");
		}
		if(trainingSet[0][0].getNumRows() != inputs)
		{
			throw new IllegalArgumentException("Training set's inputs must equal the network's input dimension!");
		}
		/*if(trainingSet[0][1].getNumRows() != outputs)
		{
			throw new IllegalArgumentException("Training set's outputs must equal the network's output dimension!");
		}*/
		
		for(int i = 0; i < trainingSet.length; i++)
		{
			model[0].activation(trainingSet[i][0]);
			for(int j = 1; j < model.length; j++)
			{
				model[j].activation(null);
			}
			
			avgCost += cost.cost(model[model.length - 1].getLastActivation(), trainingSet[i][1]);
			SimpleMatrix backin = cost.derivative(model[model.length - 1].getLastActivation(), trainingSet[i][1]);
			model[model.length - 1].reportGradient(backin);
			for(int j = model.length - 1; j >= 0; j--)
			{
				model[j].backprop();
			}
		}		
		return avgCost / trainingSet.length;
	}
	
	public SimpleMatrix feedForward(SimpleMatrix input)
	{
		SimpleMatrix out = input;
		for(int i = 0; i < model.length; i++)
		{
			out = model[i].activation(out);
		}
		return out;
	}
	
	public double getLearningRate() {
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
	
	public String stringify()
	{
		String out = model.length + "\n";
		for(Layer layer : model)
		{
			out += "||" + layer.className() + "||\n";
			out += layer.stringify();
		}
		return out;
	}
	
	public static LayersNetwork load(String model)
	{
		Scanner scanner = new Scanner(model);
		LayersNetwork layers = new LayersNetwork();
		layers.model = new Layer[scanner.nextInt()];
		scanner.useDelimiter("\\|\\|");
		for(int i = 0; i < layers.model.length; i++)
		{
			String type = scanner.next();
			String load = scanner.next();
			Layer layer;
			switch(type)
			{
			case "Standard":
				layer = new StandardLayer(0, 0, null).load(load, layers, i);
				break;
			case "Rotation":
				layer = new RotationLayer(null).load(load, layers, i);
				break;
			default:
				throw new IllegalArgumentException("Unknown layer type: " + type);
			}
			layer.setModel(layers);
			layers.model[i] = layer;
		}
		
		scanner.close();
		
		return layers;
	}
	
	public boolean[] getMasks() {
		return masks;
	}
	
	public void setMasks(boolean[] masks) {
		this.masks = masks;
	}
	
	public Layer[] getLayers()
	{
		return model;
	}
	
	public int getNextID()
	{
		return layersList.size();
	}
	
	public void reportLayer(Layer layer)
	{
		if(!layersList.contains(layer))
			layersList.add(layer);
	}
	
	public Layer getLayerByID(int id)
	{
		for(Layer layer : layersList)
			if(layer.getId() == id)
				return layer;
		throw new IllegalStateException("No layer with ID " + id + " exists!");
	}
}
