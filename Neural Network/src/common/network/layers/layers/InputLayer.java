package common.network.layers.layers;

import java.util.Scanner;

import org.ejml.simple.SimpleMatrix;

import common.network.layers.models.LayersNetwork;

public class InputLayer extends Layer{

	public InputLayer(int inputs) {
		super(inputs, inputs);
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		this.lastActivation = input;
		return input;
	}

	@Override
	public void backprop() {}
	
	@Override
	public String name() {
		return "Input";
	}
	
	@Override
	public String stringify() {
		return getId() + " " + inputs;
	}

	public static InputLayer load(String string, LayersNetwork model, int position) {
		Scanner scanner = new Scanner(string);
		int id = scanner.nextInt();
		int inputs = scanner.nextInt();
		scanner.close();
		
		InputLayer out = new InputLayer(inputs);
		out.setId(id);
		return out;
	}
}
