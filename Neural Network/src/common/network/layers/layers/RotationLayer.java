package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersNetwork;

public class RotationLayer extends Layer{

	public RotationLayer(Layer last) {
		super(last, last.depth);
		depth = last.outputs;
		lastActivation = new SimpleMatrix(outputs, depth);
		setGradientSize(outputs, depth);
	}
	
	private RotationLayer(int outputs, int depth)
	{
		super(depth, outputs);
		this.depth = depth;
		lastActivation = new SimpleMatrix(outputs, depth);
		setGradientSize(outputs, depth);
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		masks = lastLayer.getMasks();
		lastActivation = lastLayer.getLastActivation().transpose();
		return lastActivation;
	}

	@Override
	public void backprop() {
		SimpleMatrix nextErrorWeighted = getGradient();
		clearGradients();
		
		lastLayer.reportGradient(nextErrorWeighted.transpose());
	}

	@Override
	public String name() {
		return "Rotate";
	}
	
	@Override
	public String stringify() {
		return getId() + " " + lastLayer.getId() + " " + outputs + " " + depth + "\n"; 
	}
	
	@Override
	public RotationLayer load(String string, LayersNetwork model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int lastID = scanner.nextInt();
		scanner.close();
		RotationLayer out = new RotationLayer(model.getLayerByID(lastID));
		out.setId(id);
		return out;
	}
	
	@Override
	public String className() {
		return "Rotation";
	}
}
