package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/enterprise/whitelabel/pkg/config"
	"github.com/enterprise/whitelabel/pkg/server"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

var (
	configPath string
	logger     *zap.Logger
)

func main() {
	var err error
	logger, err = zap.NewProduction()
	if err != nil {
		log.Fatal("Failed to initialize logger:", err)
	}
	defer logger.Sync()

	rootCmd := &cobra.Command{
		Use:   "whitelabel-server",
		Short: "Multi-tenant white-label platform server",
		Long:  "A high-performance multi-tenant white-label platform with customization engine",
		RunE:  runServer,
	}

	rootCmd.Flags().StringVarP(&configPath, "config", "c", "config.yaml", "Path to configuration file")

	if err := rootCmd.Execute(); err != nil {
		logger.Fatal("Failed to execute command", zap.Error(err))
	}
}

func runServer(cmd *cobra.Command, args []string) error {
	// Load configuration
	cfg, err := config.Load(configPath)
	if err != nil {
		logger.Fatal("Failed to load configuration", zap.Error(err))
	}

	// Initialize server
	srv, err := server.New(cfg, logger)
	if err != nil {
		logger.Fatal("Failed to initialize server", zap.Error(err))
	}

	// Start server in a goroutine
	go func() {
		logger.Info("Starting white-label server", 
			zap.String("address", cfg.Server.Address),
			zap.Int("port", cfg.Server.Port))
		
		if err := srv.Start(); err != nil && err != http.ErrServerClosed {
			logger.Fatal("Failed to start server", zap.Error(err))
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server...")

	// Create context with timeout for graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("Server forced to shutdown", zap.Error(err))
		return err
	}

	logger.Info("Server exited")
	return nil
}