package common.network.layers.layers;

import java.util.ArrayList;

import common.network.layers.LayersMain;
import common.network.layers.models.LayersNetwork;
import common.network.math.NetworkMath;

public abstract class Layer {
	
	public final int inputs;
	public final int outputs;
	int depth = 1;
	Layer lastLayer;
	Layer nextLayer;
	protected float[][] lastActivation;
	public LayersNetwork model;
	public boolean[][] masks;
	
	private ArrayList<float[][]> gradients;
	
	public Layer(int inputs, int outputs)
	{
		this.inputs = inputs;
		this.outputs = outputs;
		gradients = new ArrayList<>();
		masks = new boolean[outputs][depth];
	}
	
	public Layer(Layer last, Layer next)
	{
		this.lastLayer = last;
		this.nextLayer = next;
		this.inputs = last.outputs;
		this.outputs = next.inputs;
		last.setNext(this);
		gradients = new ArrayList<>();
		masks = new boolean[outputs][depth];
	}
	
	public Layer(Layer last, int outputs)
	{
		this.lastLayer = last;
		this.inputs = last.outputs;
		this.outputs = outputs;
		this.depth = last.depth;
		gradients = new ArrayList<>();
		masks = new boolean[outputs][depth];
	}
	
	public abstract float[][] activation(float[][] input);
	public abstract void backprop();
	public abstract String name();
	//public abstract String display();
	
	public void setLast(Layer lastLayer) {
		this.lastLayer = lastLayer;
	}
	
	public void setNext(Layer nextLayer) {
		this.nextLayer = nextLayer;
	}
	
	public float[][] getLastActivation() {
		if(Float.isNaN(lastActivation[0][0]))
		{
			int i = 0;
			i++;
		}
		
		return lastActivation;
	}
	
	public void setModel(LayersNetwork model) {
		this.model = model;
	}
	
	public void reportGradient(float[][] gradient)
	{
		//System.out.println("Adding gradient to " + this);
		//System.out.println(LayersMain.floatMatrixToString(gradient, 2));
		
		gradients.add(gradient);
	}
	
	public float[][] getGradient()
	{
		if(gradients.size() == 0)
		{
			throw new IllegalStateException("There are no gradients!");
		}else if(gradients.size() == 1) {
			//System.out.println("Gradient of " + this);
			//System.out.println(LayersMain.floatMatrixToString(gradients.get(0), 2));
			return gradients.get(0);
		}
		
		return NetworkMath.sum(gradients.toArray(new float[gradients.size()][outputs][depth]));
	}
	
	public void clearGradients()
	{
		gradients.clear();
	}
	
	public boolean[][] getMasks() {
		return masks;
	}
	
	@Override
	public String toString() {
		return name() + " (" + outputs + ", "+ depth + ")";
	}
}
