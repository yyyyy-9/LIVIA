function repo_root = setup_paths(reticolo_root)
% Add LIVIA MATLAB core operators (and optional RETICOLO) to path.

    repo_root = fileparts(mfilename('fullpath'));
    addpath(fullfile(repo_root, '01_core_operators', 'matlab_reticolo'));

    if nargin < 1 || isempty(reticolo_root)
        reticolo_root = getenv('RETICOLO_ROOT');
    end

    if ~isempty(reticolo_root)
        if exist(reticolo_root, 'dir')
            addpath(reticolo_root);
        else
            warning('LIVIA:ReticoloPathNotFound', ...
                'RETICOLO_ROOT directory does not exist: %s', reticolo_root);
        end
    end
end
