package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersNetwork;

public class ResidualAddition extends Layer {

	Layer residual;
	
	public ResidualAddition(Layer last, Layer residual) {
		super(last, last.outputs);
		this.residual = residual;
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		masks = lastLayer.getMasks();
		lastActivation = lastLayer.getLastActivation().plus(residual.getLastActivation());
		return lastActivation;
	}

	@Override//VERIFIED
	public void backprop() {
		SimpleMatrix gradient = getGradient();
		clearGradients();
		lastLayer.reportGradient(gradient);
		residual.reportGradient(gradient);
	}

	@Override
	public String name() {
		return "Residual Addition";
	}
	
	@Override
	public String className() {
		return "PositionalEncoding";
	}
	
	@Override
	public String stringify() {
		return getId() + " " + lastLayer.getId() + " " + residual.getId() + " " + inputs + " " + depth;
	}
	
	public static ResidualAddition load(String string, LayersNetwork model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int firstID = scanner.nextInt();
		int secondID = scanner.nextInt();
		scanner.close();
		ResidualAddition out = new ResidualAddition(model.getLayerByID(firstID), model.getLayerByID(secondID));
		out.setId(id);
		return out;
	}
}
