package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersModel;

public class RotationLayer extends Layer{

	public RotationLayer(Layer last) {
		super(last, last.depth);
		depth = last.outputs;
		lastActivation = new SimpleMatrix(outputs, depth);
		setGradientSize(outputs, depth);
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input, boolean isInference) {
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
	
	public static RotationLayer load(String string, LayersModel model, int position) {
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
