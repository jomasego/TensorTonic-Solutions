import torch

def train_tiny_micrograd_mlp(inputs, targets, weights, biases, learning_rate, steps):
    w = [wt.detach().clone() for wt in weights]
    b = [bt.detach().clone() for bt in biases]

    def forward_pass(w_list, b_list):
        acts = [inputs]
        H = inputs
        for i in range(len(w_list)):
            Z = H @ w_list[i].T + b_list[i].reshape(-1)
            H = torch.tanh(Z)
            acts.append(H)
        return acts

    with torch.no_grad():
        history = []
        for _ in range(steps):
            acts = forward_pass(w, b)
            H_final = acts[-1]
            y = targets.view(-1, 1)
            loss = ((H_final - y) ** 2).sum()
            history.append(loss.clone())        # pre-update loss

            L = len(w)
            grad_W, grad_b = [None] * L, [None] * L
            G = 2 * (H_final - y)               # (B, 1)

            for l in range(L - 1, -1, -1):      # reverse order, gradients first
                H_l = acts[l + 1]
                D = G * (1 - H_l * H_l)
                grad_W[l] = D.T @ acts[l]
                grad_b[l] = D.sum(dim=0)
                if l > 0:
                    G = D @ w[l]

            for l in range(L):                   # update AFTER all gradients known
                w[l] = w[l] - learning_rate * grad_W[l]
                b[l] = b[l] - learning_rate * grad_b[l].reshape_as(b[l])

        acts_final = forward_pass(w, b)          # fresh forward on trained params
        H_final = acts_final[-1]
        preds = H_final[:, 0]
        loss_final = ((H_final - targets.view(-1, 1)) ** 2).sum()

    return preds, loss_final, w, b, history