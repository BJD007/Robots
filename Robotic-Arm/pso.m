% Particle Swarm Optimization (PSO) Algorithm
function pso_optimization
    % Problem Definition
    CostFunction = @(x) objective_function(x);  % Cost Function
    nVar = 2;  % Number of Decision Variables
    VarSize = [1 nVar];  % Decision Variables Matrix Size
    VarMin = -10;  % Lower Bound of Variables
    VarMax = 10;  % Upper Bound of Variables

    % PSO Parameters
    MaxIt = 1000;  % Maximum Number of Iterations
    nPop = 50;  % Population Size (Swarm Size)
    w = 1;  % Inertia Coefficient
    wdamp = 0.99;  % Damping Ratio of Inertia Coefficient
    c1 = 2;  % Personal Learning Coefficient
    c2 = 2;  % Global Learning Coefficient

    % Initialization
    empty_particle.Position = [];
    empty_particle.Velocity = [];
    empty_particle.Cost = [];
    empty_particle.Best.Position = [];
    empty_particle.Best.Cost = [];
    
    particle = repmat(empty_particle, nPop, 1);
    GlobalBest.Cost = inf;

    for i = 1:nPop
        % Initialize Position
        particle(i).Position = unifrnd(VarMin, VarMax, VarSize);
        % Initialize Velocity
        particle(i).Velocity = zeros(VarSize);
        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);
        % Update Personal Best
        particle(i).Best.Position = particle(i).Position;
        particle(i).Best.Cost = particle(i).Cost;
        % Update Global Best
        if particle(i).Best.Cost < GlobalBest.Cost
            GlobalBest = particle(i).Best;
        end
    end

    % PSO Main Loop
    for it = 1:MaxIt
        for i = 1:nPop
            % Update Velocity
            particle(i).Velocity = w*particle(i).Velocity ...
                + c1*rand(VarSize).*(particle(i).Best.Position - particle(i).Position) ...
                + c2*rand(VarSize).*(GlobalBest.Position - particle(i).Position);
            % Update Position
            particle(i).Position = particle(i).Position + particle(i).Velocity;
            % Apply Bounds
            particle(i).Position = max(particle(i).Position, VarMin);
            particle(i).Position = min(particle(i).Position, VarMax);
            % Evaluation
            particle(i).Cost = CostFunction(particle(i).Position);
            % Update Personal Best
            if particle(i).Cost < particle(i).Best.Cost
                particle(i).Best.Position = particle(i).Position;
                particle(i).Best.Cost = particle(i).Cost;
                % Update Global Best
                if particle(i).Best.Cost < GlobalBest.Cost
                    GlobalBest = particle(i).Best;
                end
            end
        end
        % Damping Inertia Coefficient
        w = w * wdamp;
        % Display Iteration Information
        disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(GlobalBest.Cost)]);
    end

    % Results
    disp(['Global Best Position: ', num2str(GlobalBest.Position)]);
    disp(['Global Best Cost: ', num2str(GlobalBest.Cost)]);
end

% Objective Function
function z = objective_function(x)
    z = sum(x.^2);  % Example: Sphere Function
end
