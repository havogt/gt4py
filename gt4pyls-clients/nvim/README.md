# GT4Py language server setup

Setup https://github.com/neovim/nvim-lspconfig

Add the GT4Py language server to your config

```lua
local lspconfig = require 'lspconfig'
local configs = require 'lspconfig.configs'

if not configs.gt4py then
  configs.gt4py = {
    default_config = {
      cmd = {'python3', '-m', 'functional.ffront.language_server'};
      filetypes = {'python'};
      root_dir = function(fname)
        return lspconfig.util.find_git_ancestor(fname)
      end;
      settings = {};
    };
  }
end
```

Add `gt4py` to the list of servers that should be enabled. Maybe you have something like this already

```lua
-- nvim-cmp supports additional completion capabilities
local capabilities = vim.lsp.protocol.make_client_capabilities()
capabilities = require('cmp_nvim_lsp').update_capabilities(capabilities)

-- Enable the following language servers
local servers = { 'clangd', 'gt4py' }
for _, lsp in ipairs(servers) do
  lspconfig[lsp].setup {
    on_attach = on_attach,
    capabilities = capabilities,
  }
end
```
