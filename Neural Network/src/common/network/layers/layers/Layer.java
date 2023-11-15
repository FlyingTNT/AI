package common.network.layers.layers;

import org.ejml.simple.SimpleMatrix;
import common.network.layers.models.LayersModel;
import common.network.math.NetworkMath;

/**
 * Represents a layer in a {@link common.network.layers.models.LayersModel LayersModel}
 * @author C. Cooper
 */
public abstract class Layer {
	public final int inputs;
	public final int outputs;
	int depth = 1;
	Layer lastLayer;
	Layer nextLayer;
	protected SimpleMatrix lastActivation;
	public LayersModel model;
	public boolean[] masks;
	private SimpleMatrix gradient;
	private int id = -1;
	
	public Layer(int inputs, int outputs)
	{
		this.inputs = inputs;
		this.outputs = outputs;
		masks = new boolean[outputs];
		gradient = new SimpleMatrix(new float[outputs][depth]);
		lastActivation = new SimpleMatrix(outputs, depth);
	}
	
	public Layer(Layer last, Layer next)
	{
		this.lastLayer = last;
		this.nextLayer = next;
		this.inputs = last.outputs;
		this.outputs = next.inputs;
		last.setNext(this);
		masks = new boolean[outputs];
		gradient = new SimpleMatrix(outputs, depth);
		lastActivation = new SimpleMatrix(outputs, depth);
	}
	
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
	
	public abstract SimpleMatrix activation(SimpleMatrix input);
	public abstract void backprop();
	public abstract String name();
	public abstract String stringify();
	public String className() {return name();}
	
	public void setLast(Layer lastLayer) {
		this.lastLayer = lastLayer;
	}
	
	public void setNext(Layer nextLayer) {
		this.nextLayer = nextLayer;
	}
	
	public SimpleMatrix getLastActivation() {
		
		return lastActivation;
	}
	
	public void setModel(LayersModel model) {
		this.model = model;
		if(id == -1)
			id = model.getNextID();
		model.reportLayer(this);
	}
	
	public void reportGradient(SimpleMatrix gradient)
	{		
		//System.out.println(this);
		//gradient.print();
		//this.gradient = this.gradient.plus(NetworkMath.normalize(gradient, 1));
		this.gradient = this.gradient.plus(gradient);
	}
	
	public SimpleMatrix getGradient()
	{
		return gradient;
	}
	
	public void clearGradients()
	{
		gradient = new SimpleMatrix(outputs, depth);
	}
	
	public boolean[] getMasks() {
		return masks;
	}
	
	public void setGradientSize(int width, int depth)
	{
		gradient = new SimpleMatrix(width, depth);
	}
	
	public Layer getLastLayer() {
		return lastLayer;
	}
	
	@Override
	public String toString() {
		return name() + " (" + outputs + ", "+ depth + ")";
	}
	
	public int getId() {
		return id;
	}
	
	public void setId(int id) {
		this.id = id;
	}
}
