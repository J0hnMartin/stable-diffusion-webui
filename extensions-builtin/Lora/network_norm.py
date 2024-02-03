import network


class ModuleTypeNorm(network.ModuleType):
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        if all(x in weights.w for x in ["w_norm", "b_norm"]):
            return NetworkModuleNorm(net, weights)
<<<<<<< HEAD
=======

>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
        return None


class NetworkModuleNorm(network.NetworkModule):
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        super().__init__(net, weights)
<<<<<<< HEAD
        self.w_norm = weights.w.get("w_norm")
        self.b_norm = weights.w.get("b_norm")

    def calc_updown(self, target):
        output_shape = self.w_norm.shape
        updown = self.w_norm.to(target.device, dtype=target.dtype)
        if self.b_norm is not None:
            ex_bias = self.b_norm.to(target.device, dtype=target.dtype)
        else:
            ex_bias = None
        return self.finalize_updown(updown, target, output_shape, ex_bias)
=======

        self.w_norm = weights.w.get("w_norm")
        self.b_norm = weights.w.get("b_norm")

    def calc_updown(self, orig_weight):
        output_shape = self.w_norm.shape
        updown = self.w_norm.to(orig_weight.device, dtype=orig_weight.dtype)

        if self.b_norm is not None:
            ex_bias = self.b_norm.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            ex_bias = None

        return self.finalize_updown(updown, orig_weight, output_shape, ex_bias)
>>>>>>> cf2772fab0af5573da775e7437e6acdca424f26e
