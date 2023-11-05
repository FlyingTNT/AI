package common.network.layers.layers;

import org.ejml.simple.SimpleMatrix;

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

}
