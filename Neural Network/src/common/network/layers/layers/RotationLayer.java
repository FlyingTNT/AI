package common.network.layers.layers;

import org.ejml.simple.SimpleMatrix;

public class RotationLayer extends Layer{

	public RotationLayer(Layer last) {
		super(last, last.depth);
		depth = last.outputs;
		lastActivation = new SimpleMatrix(outputs, depth);
		setGradientSize(outputs, depth);
	}

	@Override
	public SimpleMatrix activation(SimpleMatrix input) {
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

}
