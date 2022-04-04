"use strict";

import * as net from "net";
import * as path from "path";
import { ExtensionContext, ExtensionMode, workspace } from "vscode";
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
} from "vscode-languageclient/node";

let client: LanguageClient;

function getClientOptions(): LanguageClientOptions {
    return {
        // Register the server for plain text documents
        documentSelector: [
            { scheme: "file", language: "python" },
            { scheme: "untitled", language: "python" },
        ],
        outputChannelName: "GT4Py Language Server",
        synchronize: {
            // Notify the server about file changes to '.clientrc files contain in the workspace
            fileEvents: workspace.createFileSystemWatcher("**/.clientrc"),
        },
    };
}

function startLangServerTCP(addr: number): LanguageClient {
    const serverOptions: ServerOptions = () => {
        return new Promise((resolve /*, reject */) => {
            const clientSocket = new net.Socket();
            clientSocket.connect(addr, "127.0.0.1", () => {
                resolve({
                    reader: clientSocket,
                    writer: clientSocket,
                });
            });
        });
    };

    return new LanguageClient(
		'gt4py',
        `GT4Py Language Server`,
        serverOptions,
        getClientOptions()
    );
}

function startLangServer(
    command: string,
    args: string[],
    cwd: string
): LanguageClient {
    const serverOptions: ServerOptions = {
        args,
        command,
        options: { cwd },
    };

    return new LanguageClient(command, serverOptions, getClientOptions());
}

/**
 * Resolves paths that start with a tilde to the user's home directory.
 *
 * @param  {string} filePath '~/GitHub/Repo/file.png'
 * @return {string}          '/home/bob/GitHub/Repo/file.png'
 * from https://stackoverflow.com/questions/21077670/expanding-resolving-in-node-js
 */
function resolveTilde (filePath: string) {
  const os = require('os');
  if (!filePath || typeof(filePath) !== 'string') {
    return '';
  }

  // '~/folder/path' or '~' not '~alias/folder/path'
  if (filePath.startsWith('~/') || filePath === '~') {
    return filePath.replace('~', os.homedir());
  }

  return filePath;
}

export function activate(context: ExtensionContext): void {
    if (context.extensionMode === ExtensionMode.Development) {
        // Development - Run the server manually
        client = startLangServerTCP(2087);
    } else {
        // Production - Client is going to run the server (for use within `.vsix` package)
        const cwd = path.join(__dirname, "..", "..");
        const pythonPath = workspace
            .getConfiguration("python")
            .get<string>("defaultInterpreterPath");

        if (!pythonPath) {
            throw new Error("`python.defaultInterpreterPath` is not set");
        }
		const resolvedPath = resolveTilde(pythonPath);

        client = startLangServer(resolvedPath, ["-m", "functional.language_server.language_server"], cwd);
    }

    context.subscriptions.push(client.start());
}

export function deactivate(): Thenable<void> {
    return client ? client.stop() : Promise.resolve();
}
