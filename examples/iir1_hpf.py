from iir1_lpf import IIR1LPF    # Ideally, the synthesizer should follow the import and inline the implementation.

class IIR1HPF:
    """
     A single-pole high-pass IIR filter. Difference equation:

        m[n] = m[n-1] + alpha * (x[n] - m[n-1])
        y[n] = x[n] - m[n]

    The LPF state `m` is the estimated low-frequency/DC bias.
    """

    def __init__(self, *, ALPHA: float = 2**-16):   # Alpha is an instantiation time parameter.
        # If the implementation of IIR1LPF is not available to the synthesizer, it will instantiate its Verilog module
        # blindly, but then the user has to provide the signature.
        self.lpf = IIR1LPF(ALPHA=ALPHA)

    def step(self, x: float) -> float:
        x = float(x)
        bias = self.lpf.step(x)
        return x - bias
