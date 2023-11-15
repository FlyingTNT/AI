package common.network.layers.layers;

import org.ejml.simple.SimpleMatrix;
import common.network.layers.models.LayersModel;
import common.network.math.NetworkMath;

/**
 * Represents a layer in a {@link LayersModel LayersModel}
 * <br><br>
 * All layers have 2d input and 2d output.
 * @author C. Cooper
 */
public abstract class Layer {
	public final int inputs;//The number of inputs
	public final int outputs;//The number of outputs
	int depth = 1;//The second dimension of the inputs/outputs
	Layer lastLayer;//The layer that precedes this one
	protected SimpleMatrix lastActivation;//This layer's last activation. Used for forward propagation and sometimes for backpropagation
	public LayersModel model;//The model this layer belongs to
	public boolean[] masks;//An array representing which of this layer's outputs should be masked
	private SimpleMatrix gradient;//The gradient coming into this layer during backprop.
	private int id = -1;//This layer's id
	
	/**
	 * Creates a layer with the given inputs and outputs, and depth of one.
	 * @param inputs The number of inputs of this layer
	 * @param outputs The number of outputs
	 */
	public Layer(int inputs, int outputs)
	{
		this.inputs = inputs;
		this.outputs = outputs;
		masks = new boolean[outputs];
		gradient = new SimpleMatrix(new float[outputs][depth]);
		lastActivation = new SimpleMatrix(outputs, depth);
	}
	
	/**
	 * Creates a layer whose inputs and depth equal the given layer's outputs and depth.
	 * @param last The layer that precedes this layer.
	 * @param outputs The number of outputs of this layer.
	 */
	public Layer(Layer last, int outputs)
	{
		this.lastLayer = last;
		this.inputs = last.outputs;
		this.outputs = outputs;
		this.depth = last.depth;
		masks = new boolean[outputs];
		gradient = new SimpleMatrix(outputs, depth);
		lastActivation = new SimpleMatrix(outputs, depth);
	}
	
	/**
	 * Calculates the activation of this layer and stores it in lastActivation.
	 * <br><br>
	 * Theoretically, this method takes in the last layer's activation and returns its activation. However, it generally
	 * ignores the input because in models with layers like attention and residual addition, it isn't sufficient to just
	 * pass activations to the next layer in a linear manner because one activation may need to go multiple places. To solve
	 * this, layers just keep track of where their activations come from (their lastLayer) and use that layer's 
	 * {@link Layer#getLastActivation() getLastActivation()} method to get their input. As a result, the input parameter is
	 * only used by {@link InputLayer InputLayers}. Most layers do return their activation.
	 * @param input The activation of the preceding layer.
	 * @return This layer's activation.
	 */
	public abstract SimpleMatrix activation(SimpleMatrix input);
	
	/**
	 * Performs backpropagation on this layer.
	 * <br><br>
	 * This takes no inputs and gives no outputs, for the reasons described in {@link Layer#activation(SimpleMatrix) the activation method}.
	 * Instead, each layer passes its gradients back using the {@link #reportGradient(SimpleMatrix) reportGradient} method.
	 */
	public abstract void backprop();
	
	/**
	 * The name of this type of layer.
	 * @return The name of this type of layer.
	 */
	public abstract String name();
	
	/**
	 * Creates a string that can be used to store this layer. Each layer type has a load method that can use this string
	 * to reconstruct the layer.
	 * @return A string representation of all non-static data of this layer.
	 */
	public abstract String stringify();
	
	/**
	 * The same as {@link #name() name()} except without spaces so that it can be parsed more easily.
	 * @return The name of this type of layer (no spaces).
	 */
	public String className() {return name();}
	
	/**
	 * Sets this layer's last layer.
	 * @param lastLayer The layer that precedes this layer.
	 */
	public void setLast(Layer lastLayer) {
		this.lastLayer = lastLayer;
	}
	
	/**
	 * Gets this layer's last activation.
	 * @return This layer's last activation.
	 */
	public SimpleMatrix getLastActivation() {
		return lastActivation;
	}
	
	/**
	 * Sets this layer's model.<br>
	 * If this layer doesn't have an id, also uses that model's {@link LayersModel#getNextID() getNextID()} method to set its id.
	 * @param model
	 */
	public void setModel(LayersModel model) {
		this.model = model;
		if(id == -1)//-1 is the id's default value for when it hasn't been set.
			id = model.getNextID();
		model.reportLayer(this);//Tells the model that it owns this layer (this lets the model know it can't hand out this ID again)
	}
	
	/**
	 * Gives this layer a gradient to use during backprop.
	 * <br><br>
	 * If the layer is given multiple gradients, just sums them.
	 * @param gradient The gradient coming into this layer.
	 */
	public void reportGradient(SimpleMatrix gradient)
	{		
		//System.out.println(this);
		//gradient.print();
		//this.gradient = this.gradient.plus(NetworkMath.normalize(gradient, 1));
		this.gradient = this.gradient.plus(gradient);//Adds the given gradient to this layer's gradient (gradient is cleared between backprops)
	}
	
	/**
	 * @return The gradient coming into this layer.
	 */
	public SimpleMatrix getGradient()
	{
		return gradient;
	}
	
	/**
	 * Resets this layer's gradients. Must be called between backprops so that the layer isn't using old gradients.
	 */
	public void clearGradients()
	{
		gradient = new SimpleMatrix(outputs, depth);
	}
	
	/**
	 * Gets the masks on this layer's outputs.
	 * @return The mask on this layer's outputs.
	 */
	public boolean[] getMasks() {
		return masks;
	}
	
	/**
	 * Resizes this layer's gradients. Necessary for some layers, like the rotation layer because their depth is different from the depth
	 * of the layer before them.
	 * @param width This layer's output size.
	 * @param depth This layer's depth.
	 */
	public void setGradientSize(int width, int depth)
	{
		gradient = new SimpleMatrix(width, depth);
	}
	
	/**
	 * Gets the layer that precedes this layer.
	 * @return The layer that precedes this layer.
	 */
	public Layer getLastLayer() {
		return lastLayer;
	}
	
	/**
	 * Gets a string representation of this layer, in the form: {name} ({outputs}, {depth})
	 */
	@Override
	public String toString() {
		return name() + " (" + outputs + ", "+ depth + ")";
	}
	
	/**
	 * Gets this layer's id.
	 * @return This layer's id.
	 */
	public int getId() {
		return id;
	}
	
	/**
	 * Sets this layer's id.
	 * @param id The new id.
	 */
	public void setId(int id) {
		this.id = id;
	}
}
