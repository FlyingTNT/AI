package common.network.layers.layers;

import org.ejml.simple.SimpleMatrix;

public class FlattenLayer extends Layer {

	public FlattenLayer(int inputs, int inDepth) {
		super(inputs, inputs*inDepth);
		lastActivation = new SimpleMatrix(new float[outputs][1]);
		depth = 1;
	}

	public FlattenLayer(Layer last, Layer next) {
		super(last, next);
		lastActivation = new SimpleMatrix(new float[outputs][1]);
		depth = 1;
	}

	public FlattenLayer(Layer last) {
		super(last, last.outputs * last.depth);
		lastActivation = new SimpleMatrix(new float[outputs][1]);
		depth = 1;
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
		
		lastActivation = lastLayer.getLastActivation().copy();
		lastActivation.reshape(outputs, 1);
		return lastActivation;
	}

	@Override
	public void backprop() {
		SimpleMatrix nextErrorWeighted = getGradient();	
		clearGradients();
		
		nextErrorWeighted.reshape(inputs, depth);
		lastLayer.reportGradient(nextErrorWeighted);
	}

	@Override
	public String name() {
		return "Flatten";
	}

}
