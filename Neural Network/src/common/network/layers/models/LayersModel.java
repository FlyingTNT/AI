package common.network.layers.models;

import java.util.ArrayList;
import java.util.Scanner;
import org.ejml.simple.SimpleMatrix;

import common.network.layers.Cost;
import common.network.layers.layers.Decoder;
import common.network.layers.layers.EmbeddingLayer;
import common.network.layers.layers.Encoder;
import common.network.layers.layers.InputLayer;
import common.network.layers.layers.Layer;
import common.network.layers.layers.NormLayer;
import common.network.layers.layers.PositionalEncoding;
import common.network.layers.layers.ResidualAddition;
import common.network.layers.layers.RotationLayer;
import common.network.layers.layers.StandardLayer;
import common.network.layers.layers.TransformerInput;

/**
 * A machine learning model consisting of a series of {@link Layer Layers}.
 * @author C. Cooper
 */
public class LayersModel {
	protected float learningRate;//This model's learning rate.
	protected Layer[] model;//The layers in this mode (in the order they should be called for feedforward and backprop)
	protected Cost cost;//The cost function this model should train and evaluate with.
	protected int inputs;//The number of inputs to this model.
	protected int outputs;//The number of outputs
	private boolean[] masks;//Whether each input should be masked
	private ArrayList<Layer> layersList;//A list of the layers in this model, used for getting layers by their id.
	
	/**
	 * A barebones constructor used when you need more control over the internal values of the object, like with the {@link #load(String)}
	 * method.
	 */
	public LayersModel() {layersList = new ArrayList<>();}
	
	/**
	 * Creates a model with the given learning rate, cost, and sequence of layers.
	 * @param learningRate The learning rate to be used in training.
	 * @param cost The cost function to evaluate this model with.
	 * @param layers The sequence of layers that makes up this model.
	 */
	public LayersModel(float learningRate, Cost cost, Layer... layers) {
		layersList = new ArrayList<>();
		this.learningRate = learningRate;
		this.model = layers;
		this.cost = cost;
		this.inputs = layers[0].inputs;
		this.outputs = layers[layers.length - 1].outputs;
		for(int i = 0; i < layers.length; i++)
		{
			layers[i].setModel(this);
		}
		
		masks = new boolean[layers[0].inputs];
	}
	
	/**
	 * Runs an epoch on the model with the given dataset. 
	 * @param trainingSet The dataset to learn from. It should be a n x 2 array of SimpleMatricies, where n is the number of
	 * 					  input, target pairs in the dataset, and the array at each index is {inputMatrix, targetMatrix}.
	 * @return The average cost over the epoch.
	 */
	public float epoch(SimpleMatrix[]... trainingSet)
	{
		float costSum = 0;//Will hold the running sum of the cost
		if(trainingSet[0].length != 2)
		{
			throw new IllegalArgumentException("Training set's second dimension must be 2!");
		}
		if(trainingSet[0][0].getNumRows() != inputs)
		{
			throw new IllegalArgumentException("Training set's inputs must equal the network's input dimension!");
		}
		
		for(int i = 0; i < trainingSet.length; i++)//For each pair in the dataset,
		{
			model[0].activation(trainingSet[i][0], false);//Gives the input to the first layer in the model.
			for(int j = 1; j < model.length; j++)//For each subsequent layer,
			{
				model[j].activation(null, false);//Has it calculate its activation (all layer types except Input ignore the parameter in their activation method)
			}
			
			costSum += cost.cost(model[model.length - 1].getLastActivation(), trainingSet[i][1]);//Calculates the cost of the output given the target
			SimpleMatrix backin = cost.derivative(model[model.length - 1].getLastActivation(), trainingSet[i][1]);//Calculates the error of the output
			model[model.length - 1].reportGradient(backin);//Hands the error to the last layer in the model
			for(int j = model.length - 1; j >= 0; j--)//For each layer in the model,
			{
				model[j].backprop();//Backprops through that layer
			}
		}		
		return costSum / trainingSet.length;//Return the average cost
	}
	
	/**
	 * Runs inference (feedforward) on a given input.
	 * @param input The input to be fed forward.
	 * @return The output of the model.
	 */
	public SimpleMatrix feedForward(SimpleMatrix input)
	{
		SimpleMatrix out = input;
		for(int i = 0; i < model.length; i++)//For each layer,
		{
			out = model[i].activation(out, true);//Gives that layer the previous layer's activation and stores it activation.
		}
		return out;//Returns the last activation.
	}
	
	/**
	 * Gets the maodel's learning rate.
	 * @return The model's learning rate.
	 */
	public double getLearningRate() {
		return learningRate;
	}
	
	/**
	 * Creates a String representation of the model in the form: <br>
	 * "Model: {inputs} -> {outputs}<br>
	 * Layer1.toString()<br>
	 * Layer2.toString()<br>
	 * ..."
	 */
	@Override
	public String toString() {
		String out = "Model: " + inputs + " -> " + outputs + "\n";
		for(int i = 0; i < model.length; i++)
		{
			out += model[i] + "\n";
		}
		return out;
	}
	
	/**
	 * Creates a String that can be used by the {@link #load(String)} function to recreate this model.
	 * <br><br>
	 * Takes the form:<br>
	 * Model {number of layers}<br>
	 * ||Layer1.{@link Layer#className() className()}||<br>
	 * Layer1.{@link Layer#stringify() stringify()}<br>
	 * ||Layer2.{@link Layer#className() className()}||<br>
	 * ...
	 * @return A String that can be used by the {@link #load(String)} function to recreate this model.
	 */
	public String stringify()
	{
		String out = "Model " + model.length + "\n";
		for(Layer layer : model)
		{
			out += "||" + layer.className() + "||\n";
			out += layer.stringify();
		}
		return out;
	}
	
	/**
	 * Loads a model given a String produced by {@link #stringify()}
	 * @param model A string produced by {@link #stringify()}
	 * @return A model based on the given string.
	 */
	public static LayersModel load(String model)
	{
		Scanner scanner = new Scanner(model);
		LayersModel layers = new LayersModel();
		String name = scanner.next();
		if(!name.equals("Model"))//Currently, this is either "Model" or "Transformer". Just makes sure you don't try to load a Transformer with this method.
		{
			scanner.close();
			throw new IllegalArgumentException("Type is not model!: " + name);
		}
		layers.model = new Layer[scanner.nextInt()];//The next item in the string is the number of layers. Makes an array with that size.
		scanner.useDelimiter("\\|\\|");//Changes the delimiter to "||". This is the layer delimiter used in stringify()
		scanner.next();//Clears the next item because the string starts with a "||" but this section doesn't actually have any data.
		for(int i = 0; i < layers.model.length; i++)//For each layer in the number of layers (each layer is like: "type||data||")
		{
			String type = scanner.next();//Gets its type
			String load = scanner.next();//Gets its data
			System.out.println(type);//Prints the type
			System.out.println(load);//Prints the data
			Layer layer;//Variable that will store the layer
			switch(type)//Calls the relevant load function based on the type.
			{
			case "Standard":
				layer = StandardLayer.load(load, layers, i);
				break;
			case "Rotation":
				layer = RotationLayer.load(load, layers, i);
				break;
			case "Decoder":
				layer = Decoder.load(load, layers, i);
				break;
			case "Embedding":
				layer = EmbeddingLayer.load(load, layers, i);
				break;
			case "Encoder":
				layer = Encoder.load(load, layers, i);
				break;
			case "Input":
				layer = InputLayer.load(load, layers, i);
				break;
			case "Norm":
				layer = NormLayer.load(load, layers, i);
				break;
			case "PositionalEncoding":
				layer = PositionalEncoding.load(load, layers, i);
				break;
			case "ResidualAddition":
				layer = ResidualAddition.load(load, layers, i);
				break;
			case "TransformerInput":
				layer = TransformerInput.load(load, layers, i);
				break;
			default:
				throw new IllegalArgumentException("Unknown layer type: " + type);
			}
			layer.setModel(layers);//Sets the layer's model to the model we are constructing.
			layers.model[i] = layer;//Adds this layer to the array of layers.
		}
		
		scanner.close();
		
		return layers;//Returns the model we built
	}
	
	/**
	 * Gets the masks for each of this model's inputs
	 * @return The masks for each of this model's inputs
	 */
	public boolean[] getMasks() {
		return masks;
	}
	
	/**
	 * Sets the input masks.
	 * @param masks The array to set masks to.
	 */
	public void setMasks(boolean[] masks) {
		this.masks = masks;
	}
	
	/**
	 * Gets the array of layers that make up this model.
	 * @return This moden's layers.
	 */
	public Layer[] getLayers()
	{
		return model;
	}
	
	/**
	 * Gets the next available layer id.
	 * @return The next available layer id.
	 */
	public int getNextID()
	{
		return layersList.size();//Just returns the number of layers because layers are ID'd sequentially.
	}
	
	/**
	 * Adds the given layer to this model's list of layers (if the layer isn't already in the list)
	 * <br><br>
	 * The list is just used for ID purposes. This has no bearing on the array of layers that makes up this model.
	 * @param layer The layer to add.
	 */
	public void reportLayer(Layer layer)
	{
		if(!layersList.contains(layer))
			layersList.add(layer);
	}
	
	/**
	 * Given an id, returns the layer in this model's list of layers with that id.
	 * <br><br>
	 * This is used in many layers' load functions, because in non-sequential models, like transformers where the inputs
	 * to one layer do not necessarily come from the layer immediately preceding it, the layers need a way to identify what
	 * layers they are connected to. 
	 * @param id The id to search for.
	 * @return The layer with the given id.
	 * @throws IllegalStateException If there is no layer with the given id.
	 */
	public Layer getLayerByID(int id)
	{//We can't just do layersList.get(id) because layersList is often unsorted or may have gaps because layers are not always loaded sequentially.
		for(Layer layer : layersList)
			if(layer.getId() == id)
				return layer;
		throw new IllegalStateException("No layer with ID " + id + " exists!");
	}
}
