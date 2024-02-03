
import network

class ModuleTypeGLora(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["a1.weight", "a2.weight", "alpha", "b1.weight", "b2.weight"]):
            return NetworkModuleGLora(net, weights)
<<<<<<< HEAD
        return None

# adapted from https://github.com/KohakuBlueleaf/LyCORIS
class NetworkModuleGLora(network.NetworkModule): # pylint: disable=abstract-method
=======

        return None

# adapted from https://github.com/KohakuBlueleaf/LyCORIS
class NetworkModuleGLora(network.NetworkModule):
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)

        if hasattr(self.sd_module, 'weight'):
            self.shape = self.sd_module.weight.shape

        self.w1a = weights.w["a1.weight"]
        self.w1b = weights.w["b1.weight"]
        self.w2a = weights.w["a2.weight"]
        self.w2b = weights.w["b2.weight"]

<<<<<<< HEAD
    def calc_updown(self, target): # pylint: disable=arguments-differ
        w1a = self.w1a.to(target.device, dtype=target.dtype)
        w1b = self.w1b.to(target.device, dtype=target.dtype)
        w2a = self.w2a.to(target.device, dtype=target.dtype)
        w2b = self.w2b.to(target.device, dtype=target.dtype)
        output_shape = [w1a.size(0), w1b.size(1)]
        updown = (w2b @ w1b) + ((target @ w2a) @ w1a)
        return self.finalize_updown(updown, target, output_shape)
=======
    def calc_updown(self, orig_weight):
        w1a = self.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
        w1b = self.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
        w2a = self.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
        w2b = self.w2b.to(orig_weight.device, dtype=orig_weight.dtype)

        output_shape = [w1a.size(0), w1b.size(1)]
        updown = ((w2b @ w1b) + ((orig_weight @ w2a) @ w1a))

        return self.finalize_updown(updown, orig_weight, output_shape)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
